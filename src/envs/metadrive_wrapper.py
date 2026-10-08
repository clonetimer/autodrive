"""Real MetaDrive adapter; never substitutes native dynamics on import failure.

Audited against the user-supplied MetaDrive 0.4.3 sources. NativeGym and
MetaDrive have identical local observation schemas but DIFFERENT physics and
rewards. Real simulator integration tests require Panda3D and matching assets.
"""
from __future__ import annotations
from dataclasses import replace
import importlib.util
import ast
import logging
import math
from pathlib import Path
import numpy as np
from src.envs.observation import Snapshot, build_ego_motion, build_observation, build_state, observation_space, state_dimension
from src.envs.reward import reward_terms, traffic_risk
from src.scenarios.traffic import Vehicle
from src.utils.config import EnvConfig
from src.utils.spaces import Box, Env


class MetaDriveUnavailable(RuntimeError):
    """Missing real simulator prerequisites. Training must fail loudly."""


def prerequisite_status(source: str | None = None) -> dict:
    required = ['gymnasium', 'panda3d', 'pygame', 'metadrive', 'progressbar', 'shapely', 'gltf', 'mediapy', 'yapf']
    packages = {}
    for name in required:
        try:
            packages[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            packages[name] = False
    assets, source_version, assets_version = None, None, None
    source_root = Path(source) / 'metadrive' if source else None
    if source_root is None and packages['metadrive']:
        spec = importlib.util.find_spec('metadrive')
        source_root = Path(spec.origin).parent if spec.origin else None
    if source_root is not None:
        candidate = source_root / 'assets' / 'version.txt'
        assets = candidate.is_file()
        if assets:
            assets_version = candidate.read_text(encoding='utf-8').strip()
        version_file = source_root / 'version.py'
        if version_file.is_file():
            try:
                for node in ast.parse(version_file.read_text(encoding='utf-8')).body:
                    if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == 'VERSION' for n in node.targets):
                        source_version = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                source_version = None
    matches = bool(source_version is not None and source_version == assets_version)
    return {'packages': packages, 'assets_present': assets, 'source_version': source_version,
            'assets_version': assets_version, 'assets_version_matches': matches,
            'ready': all(packages.values()) and assets is True and matches,
            'reason': 'Requires real importable dependencies and version-matched assets; runtime smoke is a separate gate.'}


def validate_action(action) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError('MetaDrive action is [steering, signed_throttle_brake], two finite values')
    return np.clip(array, -1, 1)


def steering_radians(normalized: float, maximum_degrees: float) -> float:
    """MetaDrive 0.4.3 _set_action scales normalized steer by max_steering degrees."""
    if not np.isfinite([normalized, maximum_degrees]).all() or maximum_degrees <= 0:
        raise ValueError('Steering and its degree limit must be finite and valid')
    return float(np.deg2rad(np.clip(normalized, -1, 1) * maximum_degrees))


def simulator_config(config: EnvConfig) -> dict:
    """Build only actual backend-supported keys; ScenarioEnv != MetaDriveEnv."""
    result = dict(use_render=False, image_observation=(config.observation == 'rgb_ego'), map=config.metadrive_map,
        start_seed=0, num_scenarios=100000, horizon=config.max_steps,
        traffic_density=config.metadrive_traffic_density, log_level=logging.ERROR,
        truncate_as_terminate=False, physics_world_step_size=config.dt / config.substeps,
        decision_repeat=config.substeps, out_of_road_done=True,
        vehicle_config=dict(lidar=dict(num_lasers=72, distance=50, num_others=4)))
    if config.dataset:
        if config.num_agents != 1:
            raise ValueError('ScenarioEnv dataset replay currently supports a single learning ego')
        for key in ('map', 'traffic_density', 'start_seed', 'out_of_road_done'):
            result.pop(key)
        # ScenarioEnv rejects unknown keys and uses its own off-road termination.
        result.update(data_directory=str(Path(config.dataset).resolve()), num_scenarios=-1,
            crash_vehicle_done=True, crash_object_done=True, crash_human_done=True,
            start_scenario_index=0, reactive_traffic=False)
    elif config.num_agents > 1:
        result.update(num_agents=config.num_agents, allow_respawn=False, delay_done=0,
                      crash_done=True, force_seed_spawn_manager=True)
    return result


class MetaDriveParallelEnv:
    """Shared real simulator with no respawn; per-agent terminations preserved.

    A MetaDrive process owns one Panda3D engine. Always close this instance before
    creating another in the same process. Multi-process workers may each own one.
    """
    def __init__(self, config: EnvConfig):
        self.config = config
        missing = [key for key, available in prerequisite_status()['packages'].items() if not available]
        if missing:
            raise MetaDriveUnavailable('Missing dependencies: ' + ', '.join(missing) +
                '. Run scripts/bootstrap_metadrive.py in a network-enabled environment or provide offline wheels. No fallback was used.')
        try:
            from metadrive import MetaDriveEnv, MultiAgentMetaDrive, ScenarioEnv
        except ImportError as exc:
            raise MetaDriveUnavailable(f'MetaDrive import failed: {exc}') from exc
        if not prerequisite_status()['assets_present']:
            raise MetaDriveUnavailable('MetaDrive assets missing; run python -m metadrive.pull_asset before training.')
        if not prerequisite_status()['assets_version_matches']:
            raise MetaDriveUnavailable('MetaDrive assets/source version mismatch; reinstall matching assets, no fallback was used.')
        md_config = simulator_config(config)
        if config.observation == 'rgb_ego':
            if config.num_agents != 1:
                raise ValueError('rgb_ego currently supports a single MetaDrive learning ego')
            from metadrive.component.sensors.rgb_camera import RGBCamera
            md_config.update(image_observation=True, image_on_cuda=False, stack_size=1, show_terrain=False,
                             show_interface=False, show_logo=False, show_fps=False,
                             sensors={'rgb_camera': (RGBCamera, config.rgb_width, config.rgb_height)})
            md_config['vehicle_config'] = dict(md_config.get('vehicle_config', {}), image_source='rgb_camera')
        self.multi = config.num_agents > 1
        if config.dataset:
            if self.multi:
                raise ValueError('ScenarioEnv dataset replay currently supports a single learning ego')
            self.env = ScenarioEnv(md_config)
        elif self.multi:
            # Capture physical terminal states BEFORE MetaDrive destroys/removes
            # finished vehicles. Retaining a Python vehicle reference is not enough.
            class CaptureMultiAgent(MultiAgentMetaDrive):
                capture_callback = None

                def _after_vehicle_done(inner, obs, rewards, terms, truncs, infos):
                    if inner.capture_callback is not None:
                        inner.capture_callback()
                    return super()._after_vehicle_done(obs, rewards, terms, truncs, infos)

            self.env = CaptureMultiAgent(md_config)
            self.env.capture_callback = self._capture_terminal_snapshots
        else:
            self.env = MetaDriveEnv(md_config)
        self.possible_agents = [f'agent{i}' for i in range(config.num_agents)]
        self.agents = []
        self._raw_ids = {}
        self.rng = np.random.default_rng(config.seed)
        self.action_space = Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = observation_space(config.observation, config.neighbors, config.bev_size, config.bev_frames, config.rgb_width, config.rgb_height)
        self.global_state_dim = config.num_agents * (state_dimension(config.neighbors) + 1)
        self.steps = 0
        self.histories = {}
        self._last_snapshots = {}
        self._speed = {}
        self._acceleration = {}
        self._completion = {}
        self._seed = config.seed
        self._raw_rgb = {}

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._seed = seed
        # Always choose a procedural seed explicitly for predictable seed domains.
        scenario_seed = int(seed if seed is not None else self.rng.integers(0, 1000))
        if self.config.dataset and seed is None:
            raw_reset_obs, _ = self.env.reset()  # ScenarioEnv chooses a legal loaded dataset index.
            scenario_seed = int(self.env.current_seed)
        else:
            raw_reset_obs, _ = self.env.reset(seed=scenario_seed)
        self._seed = scenario_seed
        raw_ids = sorted(self.env.agents.keys())
        if len(raw_ids) != len(self.possible_agents):
            raise RuntimeError('MetaDrive did not spawn the requested fixed learning population')
        self._raw_ids = dict(zip(self.possible_agents, raw_ids))
        self.agents = self.possible_agents.copy()
        self.steps = 0
        self._speed, self._acceleration, self._completion = {}, {}, {}
        self._lateral_acceleration, self._previous_steering = {}, {}
        self._last_snapshots = {name: self._snapshot(name) for name in self.agents}
        from src.perception.history import BEVHistory
        semantic_modes = {'bev', 'hybrid', 'bev_ego'}
        self.histories = {name: BEVHistory(self.config.bev_frames, self.config.bev_size, self.config.dt)
                          for name in self.agents} if self.config.observation in semantic_modes else {}
        for name in self.histories:
            self.histories[name].capture(self._last_snapshots[name], 0.0)
        self._raw_rgb = {}
        self._safety_supervisors = {}
        if self.config.safety_supervisor:
            from src.safety.supervisor import SafetySupervisor
            self._safety_supervisors = {
                name: SafetySupervisor(ttc_threshold_s=self.config.safety_ttc_threshold_s,
                                       risk_threshold=self.config.safety_risk_threshold)
                for name in self.agents
            }
        if self.config.observation == 'rgb_ego':
            self._raw_rgb['agent0'] = self._rgb_from_raw(raw_reset_obs)
        self._returns = {name: 0.0 for name in self.agents}
        self._official_returns = {name: 0.0 for name in self.agents}
        for name, snapshot in self._last_snapshots.items():
            self._speed[name] = snapshot.ego.speed
            self._acceleration[name] = 0.0
            self._completion[name] = snapshot.completion
            self._lateral_acceleration[name] = 0.0
            self._previous_steering[name] = snapshot.ego.steer
        return ({name: self.observe(name) for name in self.agents},
                {name: {'backend': 'metadrive', 'scenario_seed': scenario_seed} for name in self.agents})

    def _vehicle(self, raw, name: str) -> Vehicle:
        # MetaDrive BaseObject.speed is m/s (speed_km_h is a separate property).
        return Vehicle(name, float(raw.position[0]), float(raw.position[1]), float(raw.heading_theta),
            float(raw.speed), self.config.target_speed, float(raw.LENGTH), float(raw.WIDTH),
            acceleration=self._acceleration.get(name, 0.0),
            steer=steering_radians(float(getattr(raw, 'steering', 0)), float(raw.max_steering)),
            learning=name in self.possible_agents)

    def _snapshot(self, name: str, raw_override=None) -> Snapshot:
        raw = raw_override if raw_override is not None else self.env.agents[self._raw_ids[name]]
        ego = self._vehicle(raw, name)
        navigation = raw.navigation
        lanes = list(navigation.current_ref_lanes or [raw.lane])
        lane = raw.lane
        longitudinal, lateral = lane.local_coordinates(raw.position)
        width = float(lane.width_at(longitudinal))
        lane_index = lanes.index(lane) if lane in lanes else 0
        # Sample actual lane geometry, rather than a straight-road approximation.
        samples = np.linspace(max(-12.0, longitudinal - 20), min(float(lane.length), longitudinal + 80), 45)
        boundaries = [np.asarray([ref.position(float(s), -ref.width_at(s) / 2) for s in samples]) for ref in lanes]
        boundaries.append(np.asarray([lanes[-1].position(float(s), lanes[-1].width_at(s) / 2) for s in samples]))
        route = np.asarray([lane.position(float(s), 0) for s in samples])
        if navigation.next_ref_lanes:
            next_lane = navigation.next_ref_lanes[min(lane_index, len(navigation.next_ref_lanes) - 1)]
            extra = np.asarray([next_lane.position(float(s), 0) for s in np.linspace(0, min(60, next_lane.length), 20)])
            route = np.concatenate((route, extra))
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        objects = self.env.engine.get_objects(lambda obj: isinstance(obj, BaseVehicle)).values()
        # MetaDrive's ``env.agents`` keys are policy/agent handles while
        # ``BaseObject.name`` is an independent object UUID. Mapping via the
        # object name therefore duplicated the learning ego in Snapshot. That
        # duplicate sat at dx=0 and made geometry-derived risk report a
        # permanent front collision. Map the *actual Python vehicle object* to
        # our logical agent name instead. This also survives MetaDrive versions
        # whose raw agent handle is ``default_agent`` rather than the object UUID.
        learning_objects = {}
        for logical, raw_id in self._raw_ids.items():
            raw_agent = self.env.agents.get(raw_id)
            if raw_agent is not None:
                learning_objects[id(raw_agent)] = logical
        cars = [self._vehicle(obj, learning_objects.get(id(obj), str(obj.name))) for obj in objects]
        # Replace any engine copy of this ego with the canonical object built
        # from ``raw`` so the snapshot contains exactly one logical ego.
        cars = [v for v in cars if v.id != name] + [ego]
        return Snapshot(ego, cars, boundaries, route, float(lane.heading_theta_at(longitudinal)),
            float(lateral), float(lateral + (lane_index - (len(lanes) - 1) / 2) * width), width,
            len(lanes), lane_index, float(np.clip(navigation.route_completion, 0, 1)))

    def _capture_terminal_snapshots(self):
        self._captured_step = {name: self._snapshot(name) for name in self.agents}

    def _rgb_from_raw(self, raw_observation) -> np.ndarray:
        if not isinstance(raw_observation, dict) or 'image' not in raw_observation:
            raise RuntimeError('MetaDrive rgb_ego requested but RGBCamera image is missing from observation')
        image = np.asarray(raw_observation['image'])
        if image.ndim == 4:
            if image.shape[-1] != 1:
                raise RuntimeError(f'Expected one RGB stack frame, got shape {image.shape}')
            image = image[..., 0]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise RuntimeError(f'Unexpected MetaDrive RGB image shape {image.shape}')
        if np.issubdtype(image.dtype, np.floating):
            if not np.isfinite(image).all():
                raise FloatingPointError('MetaDrive camera returned non-finite pixels')
            image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
        expected = (self.config.rgb_height, self.config.rgb_width, 3)
        if image.shape != expected:
            raise RuntimeError(f'MetaDrive RGB shape {image.shape} does not match configured {expected}')
        return np.ascontiguousarray(image.transpose(2, 0, 1))

    def observe(self, name):
        if self.config.observation == 'rgb_ego':
            return {'rgb': self._raw_rgb[name].copy(),
                    'motion': build_ego_motion(self._last_snapshots[name], self.config.state_schema_version)}
        return build_observation(self._last_snapshots[name], self.config, self.rng, self.histories.get(name))

    def global_state(self):
        pieces = []
        for name in self.possible_agents:
            valid = name in self.agents
            features = build_state(self._last_snapshots[name], self.config.neighbors, self.config.state_schema_version) if valid else np.zeros(state_dimension(self.config.neighbors))
            pieces.extend(features)
            pieces.append(float(valid))
        return np.asarray(pieces, dtype=np.float32)

    def step(self, actions):
        active = self.agents.copy()
        if not active or set(actions) != set(active):
            raise ValueError('Actions must match active agents; reset after completion')
        proposed_actions = {name: validate_action(actions[name]) for name in active}
        route_decisions = {}
        route_actions = proposed_actions
        if self.config.route_assist:
            from src.navigation.route_assist import RouteAssist
            route_decisions = {}
            route_actions = {}
            for name in active:
                state = build_state(self._last_snapshots[name], self.config.neighbors, self.config.state_schema_version)
                decision = RouteAssist(
                    self.config.route_assist_blend,
                    mode=self.config.route_assist_mode,
                    min_blend=self.config.route_assist_min_blend,
                    curvature_start=self.config.route_assist_curvature_start,
                    curvature_full=self.config.route_assist_curvature_full,
                    heading_start_rad=self.config.route_assist_heading_start_rad,
                    heading_full_rad=self.config.route_assist_heading_full_rad,
                    lane_start=self.config.route_assist_lane_start,
                    lane_full=self.config.route_assist_lane_full,
                ).decide(proposed_actions[name], state)
                route_decisions[name] = decision
                route_actions[name] = decision.applied
        safety_decisions = {}
        if self.config.safety_supervisor:
            for name in active:
                safety_decisions[name] = self._safety_supervisors[name].decide(
                    route_actions[name], self._last_snapshots[name])
        applied_actions = {name: (safety_decisions[name].applied if name in safety_decisions else route_actions[name])
                           for name in active}
        validated = {self._raw_ids[name]: applied_actions[name] for name in active}
        # Save references: the real multi-agent environment removes done vehicles.
        raw_refs = {name: self.env.agents[self._raw_ids[name]] for name in active}
        raw_o, raw_r, raw_t, raw_u, raw_i = self.env.step(validated if self.multi else validated[self._raw_ids['agent0']])
        if not self.multi:
            key = self._raw_ids['agent0']
            raw_observations = {key: raw_o}
            raw_r, raw_t, raw_u, raw_i = ({key: value} for value in (raw_r, raw_t, raw_u, raw_i))
        else:
            raw_observations = raw_o
        self.steps += 1
        obs, rewards, terms, truncs, infos = {}, {}, {}, {}, {}
        for name in active:
            key = self._raw_ids[name]
            info = dict(raw_i.get(key, {}))
            previous_snapshot = self._last_snapshots[name]
            snapshot = self._captured_step[name] if self.multi else self._snapshot(name, raw_refs[name])
            if self.config.observation == 'rgb_ego':
                self._raw_rgb[name] = self._rgb_from_raw(raw_observations[key])
            speed = snapshot.ego.speed
            acceleration = (speed - self._speed[name]) / self.config.dt
            jerk = (acceleration - self._acceleration[name]) / self.config.dt
            self._speed[name], self._acceleration[name] = speed, acceleration
            from src.scenarios.traffic import wrap_angle
            snapshot.ego.yaw_rate = wrap_angle(snapshot.ego.yaw - self._last_snapshots[name].ego.yaw) / self.config.dt
            lateral_acceleration = speed * snapshot.ego.yaw_rate
            lateral_jerk = (lateral_acceleration - self._lateral_acceleration[name]) / self.config.dt
            steering_rate = (snapshot.ego.steer - self._previous_steering[name]) / self.config.dt
            self._lateral_acceleration[name] = lateral_acceleration
            self._previous_steering[name] = snapshot.ego.steer
            snapshot.ego.acceleration = acceleration
            if name in self.histories:
                self.histories[name].capture(snapshot, self.steps * self.config.dt)
            self._last_snapshots[name] = snapshot
            crash = bool(info.get('crash_vehicle', False) or info.get('crash_object', False) or info.get('crash_building', False) or info.get('crash_human', False))
            offroad = bool(info.get('out_of_road', False))
            success = bool(info.get('arrive_dest', False)) and not crash and not offroad
            terminated = bool(raw_t.get(key, False))
            truncated = bool(raw_u.get(key, False)) and not terminated
            # Simulator horizon is authoritative; no fabricated success fallback.
            completion_delta = snapshot.completion - self._completion[name]
            displacement = np.array([snapshot.ego.x - previous_snapshot.ego.x,
                                     snapshot.ego.y - previous_snapshot.ego.y], dtype=np.float64)
            tangent = np.array([math.cos(previous_snapshot.road_heading),
                                math.sin(previous_snapshot.road_heading)], dtype=np.float64)
            progress_m = float(displacement @ tangent)
            risk = traffic_risk(snapshot)
            official_reward = float(raw_r[key])
            custom_terms = reward_terms(snapshot, progress_m, jerk, self.config.dt, self.config.reward,
                                        self.config.reward_version, collision=crash, offroad=offroad,
                                        success=success, risk=risk)
            custom_reward = float(sum(custom_terms.values()))
            used_reward = official_reward if self.config.reward_source == 'official' else custom_reward
            infos[name] = dict(info, backend='metadrive', scenario_seed=self._seed,
                collision=crash, out_of_road=offroad, success=success, speed=speed,
                acceleration=acceleration, jerk=jerk,
                lateral_acceleration=lateral_acceleration, lateral_jerk=lateral_jerk,
                steering=snapshot.ego.steer, steering_rate=steering_rate,
                yaw_rate=snapshot.ego.yaw_rate, lane_offset=snapshot.lane_offset,
                lane_width=snapshot.lane_width, route_completion=snapshot.completion,
                progress=completion_delta, progress_m=progress_m, dt=self.config.dt,
                time=self.steps * self.config.dt, step=self.steps, x=snapshot.ego.x, y=snapshot.ego.y,
                yaw=snapshot.ego.yaw, lane_changes=0, reward_source=self.config.reward_source,
                reward_version=self.config.reward_version, official_reward=official_reward,
                custom_reward=custom_reward, reward_terms=custom_terms, risk=risk.to_dict(),
                safety_supervisor=self.config.safety_supervisor,
                safety_intervention=bool(safety_decisions.get(name).intervened) if name in safety_decisions else False,
                safety_reason=safety_decisions.get(name).reason if name in safety_decisions else 'disabled',
                safety_severity=float(safety_decisions.get(name).severity) if name in safety_decisions else 0.0,
                route_assist=self.config.route_assist,
                route_assist_mode=self.config.route_assist_mode if self.config.route_assist else 'disabled',
                route_assist_intervention=bool(route_decisions.get(name).intervened) if name in route_decisions else False,
                route_assist_blend=float(route_decisions.get(name).blend) if name in route_decisions else 0.0,
                route_assist_need_score=float(route_decisions.get(name).need_score) if name in route_decisions else 0.0,
                learned_steering_authority=float(1.0-route_decisions.get(name).blend) if name in route_decisions else 1.0,
                route_assist_target_steering=float(route_decisions.get(name).target_steering) if name in route_decisions else float(proposed_actions[name][0]),
                route_assist_steering_correction=float(route_decisions.get(name).correction) if name in route_decisions else 0.0,
                proposed_action=proposed_actions[name].copy(),
                route_assisted_action=route_actions[name].copy(), applied_action=applied_actions[name].copy())
            self._completion[name] = snapshot.completion
            self._official_returns[name] += official_reward
            self._returns[name] += used_reward
            infos[name]['episode_return'] = self._returns[name]
            infos[name]['official_episode_return'] = self._official_returns[name]
            rewards[name] = used_reward
            terms[name], truncs[name] = terminated, truncated
            obs[name] = self.observe(name)
        self.transition_state = self.global_state()
        for name in active:
            if terms[name] or truncs[name]:
                self.agents.remove(name)
        return obs, rewards, terms, truncs, infos

    def render(self, name='agent0'):
        from src.perception.bev_generator import rasterize, semantic_rgb
        return semantic_rgb(rasterize(self._last_snapshots[name], 192))

    def close(self):
        self.agents = []
        self.env.close()


class MetaDriveWrapper(Env):
    """Real single-agent Gymnasium facade, not the legacy pass-through stub."""
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(self, config: EnvConfig | None = None):
        self.world = MetaDriveParallelEnv(replace(config or EnvConfig(backend='metadrive'), num_agents=1))
        self.action_space = self.world.action_space
        self.observation_space = self.world.observation_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        o, i = self.world.reset(seed=seed, options=options)
        return o['agent0'], i['agent0']

    def step(self, action):
        o, r, t, u, i = self.world.step({'agent0': action})
        return o['agent0'], r['agent0'], t['agent0'], u['agent0'], i['agent0']

    def render(self):
        return self.world.render()

    def close(self):
        self.world.close()
