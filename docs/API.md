# API 约定

## 环境

`NativeWorld(EnvConfig)` / `MetaDriveParallelEnv(EnvConfig)` / `ReplayWorld(EnvConfig)`：

```python
observations, infos = env.reset(seed=2000)
while env.agents:
    actions = {name: policy.predict(observations[name]) for name in env.agents}
    next_obs, rewards, terminated, truncated, infos = env.step(actions)
    observations = {name: next_obs[name] for name in env.agents}
env.close()
```

`reset` 返回字典映射；`step` 返回五项字典，不使用 `__all__` 伪 agent。`possible_agents` 固定，`agents` 动态缩减。终态仍在该 step 的返回值中。下一 step 不得再给已终止车辆动作。无活跃车辆后必须 reset。

`SimpleDriveEnv` / `MetaDriveWrapper` 是单车五返回值 facade。有真实 Gymnasium 时继承其 Env；没有时仅适用于本项目 native trainer，不声称 Gymnasium 注册兼容。

## 旧观测 schema（兼容 v2.0）

- state：float32，长度 `20 + 7 * neighbors`，默认 62。
- bev：`{"bev": uint8[5,H,W]}`。
- hybrid：`{"state": float32[62], "bev": uint8[5,H,W]}`。

state 前 12 项：速度/30、航向误差 sin/cos、最近车道偏移/宽度、加速度/6、目标速度/30、道路横向偏移归一化、车道数/4、车道宽/5、剩余完成度、转向/0.45、车道序号归一化。

随后 4 个 route 点：局部 x/60、y/20。每个 neighbor 7 项：局部 x/60、y/20、相对 vx/30、vy/30、相对 heading sin/cos、presence mask。邻车按距离与 ID 排序，缺失项补零，观测噪声不污染 padding mask。

Actor 没有绝对世界坐标输入，但路线、交通真值和导航语义是 privileged simulation measurements。不是从图像检测出来的状态。

## 动作

`float32[2]`，依次 steering ∈[−1,1]、signed pedal ∈[−1,1]。Native 归一化转向映射最大 0.45rad，pedal 正值最大 3m/s²，负值最大制动 −6m/s²，并有执行器滞后/阻力；MetaDrive 使用其原生方向盘、油门/制动语义，不硬凑第三维 brake。非有限或维度错误动作拒绝，越界值饱和。

## 策略与训练

`train(TrainConfig, output_dir, resume=None)` 按 algorithm 分派。`load_checkpoint(path)` 返回 policy、解析配置、载入元数据。`EndToEndAgent.predict(obs)` 只做神经策略推理；`act(obs_dict)` 共享策略处理多车。模型不调用规则控制器做隐藏修正。

MAPPO critic 输入维度是 `num_agents * (state_dim + 1) + num_agents`。最后 num_agents 项是 self ID。Actor 仍然只接收单车局部观测。

## 评估

`evaluate(agent_factory, env_config, seeds, output, label, save_trajectories)` 输出统一 dict，CSV/JSON/HTML/Markdown 共用该 dict。成功必须由环境显式到达终点信号触发；timeout 不算成功。评估配置若变更后端或观测结构，调用方必须保证模型 schema 兼容。

## 数据

Portable JSON schema_version=1，tracks 每行 `[time_s,x_m,y_m,yaw_rad,speed_mps]`；要求有限数、严格递增时间、非负速度、唯一 ID、来源声明。轨迹回放车按记录状态移动，不会响应 ego，因此与 IDM 背景模型的闭环交互性质不同。

## 配置组合与训练课程

`load_config()` 支持 `include: [env/native.yaml, algorithm/ppo.yaml]`；相对路径以包含文件为基准，后合并覆盖先合并，CLI overrides 最后覆盖。检测循环 include 与未知字段。每次训练写出完整解析配置，而非只保存引用。

`configs/curriculum.yaml` 将课程接入 PPO/IPPO/MAPPO 的**联合回合结束点**。20 个近期联合场景以上，联合成功率 ≥0.75 且任何车辆碰撞率 ≤0.15 才升级。阶段为 `(0车,直道)`、`(5车,幅度2米弯道)`、`(12车,幅度5米弯道)`。仅修改训练场景；checkpoint 的 config.env 保留目标评估配置，extra 保存课程阶段/历史。恢复时还原课程，不保证仿真轨迹逐位连续。不支持 SAC/MetaDrive 课程，显式报错。阶段日志在 `curriculum.csv`，未晋级也如实记录。

`jerk_*_mps3` 保留纵向差分语义；v2.1 新增横向指标，见下节。高斯观测噪声不是图像噪声/遮挡模型。


## v2.1 观测与版本化奖励

`EnvConfig.reward_version` 取 2/3，新配置默认3；历史checkpoint未存该字段时恢复为2。`bev_frames` 为1–8整数，state只允许1。`bev_ego` 是 `{"bev": uint8[5F,H,W], "motion": float32[20]}`；F>1时另有 `history: float32[2F]`，包含归一化帧龄和有效性。bev/hybrid也支持多帧。最新帧在前，reset后历史为零且无效，重复读取不推进时间。

motion顺序以 `build_ego_motion()` 为准：速度、加速度、yaw_rate、转向、目标速度、航向误差sin/cos、车道偏移、道路偏移、车道数/宽度、剩余进度及四个路线局部点。没有邻车速度。多帧对齐使用每帧真值姿态和最近邻采样；场景终止和reset不跨帧泄漏。

`info.reward_terms` 为全部奖励项，`info.risk` 包含道路投影的前车间距/接近速度/TTC和风险目标速度；无前车时TTC为null。新增风险启发式不是CBF或安全保证，MetaDrive仍使用其原生奖励。

## v2.1 SAC 持久化

推理读取 `model.pt`；训练恢复读取 `training_state.pt` 或 `.resume.pt`。恢复检验完整配置，仅训练总目标、checkpoint频率、device/threads可调整。Replay保存已用环形槽、cursor、采样随机状态，拒绝错误形状/类型/不合法部分缓冲。`replay_max_bytes` 默认536870912，避免多帧uint8图像buffer分配超过内存限额。

恢复全部学习状态、环境随机发生器状态和计数，但不序列化车辆/模拟器物理状态。开始新回合；日志 `world_steps_total` 为累计，`world_steps_this_run` 为本段增量，不重复计数。

## v2.1 物理指标与失败证据

新增横向加速度 m/s²、横向jerk m/s³、转向率 rad/s；mean abs/RMS使用真实dt。p95为时间权重经验分位数；聚合 `mean_episode_p95` 不是全局p95。无有效测量为null并有coverage，因浮点累计可能出现极微小的1附近舍入误差。

`failure_traces.json` 默认保留所有失败rollout末4秒，包含动作、reward_terms、risk、终态；`evaluation.json`保存解析环境配置和真实后端类名。成功不因time limit而触发；文件中的场景回合和agent回合分别计数。


## v2.2 MetaDrive 离线部署与证据入口

- `src.utils.metadrive_assets.audit_upload/asset_inventory/stage_assets/verify_staged_assets`：读取真实归档内容、验证版本/引用/哈希，保护已有源码并可重复部署。
- `src.utils.installation.validate_target_base/check_installed_closure`：精确基础版本与依赖图验证，不用 wheel 标签替代真实安装验证。
- `src.envs.metadrive_contract.reference_config/reference_plan`：官方参考实验配置与互斥种子划分。
- `scripts/train_metadrive_official.py`：真实 Gymnasium/MetaDrive/SB3 CPU参考训练，关闭训练引擎后进行模型冷加载评估，无 native 后备路径。
- `scripts/inspect_scenarionet.py`：受限数值 pickle 读取和已有转换场景审计，不调用物理引擎。
- `scripts/export_dependency_request.py` 与 `tools/collect_metadrive_wheels.py`：导出目标及跨平台收集缺失 wheels；网络链路当前未实测通过。

命令与实际验证范围详见 `METADRIVE_OFFLINE_v2.2.0.md`。

## v2.3 MetaDrive 真实运行 API

### OfficialMetaDriveGymEnv

`src.envs.metadrive_official.OfficialMetaDriveGymEnv(config)` 为 MetaDrive 0.4.3 原始单车环境增加 Gymnasium 1.2 兼容 facade。`reset(seed=..., options=None)` 会把任意 Gym seed 映射到配置的 `start_seed..start_seed+num_scenarios-1`。非空 `options` 暂不支持并明确抛错。`raw_env` 暴露底层 MetaDriveEnv 用于官方指标读取；策略代码不应依赖它获得额外执行端真值。

### 分段 MetaDrive 训练

`src.utils.segmented.segment_targets(total, segment)` 生成累计目标。`scripts/train_metadrive_segmented.py` 每段用新的 Python/Panda3D 进程执行 `scripts/train.py`；仅当 `result.json`、`model.pt` 和累计世界步满足目标时接受该段。超时/返回码失败会从最后接受的模型重试。checkpoint 恢复学习状态，但物理世界从新 episode 开始。

### ScenarioEnv 数据闭环

`scripts/evaluate_metadrive_dataset.py --model ... --dataset ...` 使用模型内配置替换为 `backend=metadrive`、真实 `dataset` 路径、单学习agent，然后复用统一 `evaluate()`。输出仍是标准 `evaluation.json/csv/md/html`；另写 `dataset_evaluation.json` 记录数据目录、可用样本数和背景交通解释边界。

### RGB sensor smoke

`scripts/metadrive_camera_smoke.py` 默认由父进程启动隔离 worker。有 `xvfb-run` 时优先创建虚拟 X display，并使用 `image_on_cuda=False` 读取 CPU `RGBCamera`。父进程记录 return code/timeout/worker JSON。该 API 只验证传感器和 renderer，不返回可训练 policy。

### v2.3 发布模型

`models/registry.json` 新增四个真实 MetaDrive 模型条目：自研 PPO state 32k、MAPPO hybrid 8k、PPO bev_ego×4 4k 与官方 SB3 PPO 4k。每个条目含 SHA-256、训练世界步和对应评估路径。SB3 模型为其原生 ZIP 格式；项目自研模型为 Torch checkpoint `.pt`。

完整实测与限制见 `METADRIVE_REAL_v2.3.0.md`。

## v2.4 SafetySupervisor 与 raw RGB API

`EnvConfig.reward_version=4` 增加 TTC/closing/unsafe-acceleration/emergency-brake 项；v2/v3 仍可恢复历史模型。`info.reward_terms` 保留每项分解。

`EnvConfig.safety_supervisor=True` 仅允许真实 MetaDrive。`MetaDriveParallelEnv` 在 neural actor 输出后调用 `SafetySupervisor.decide()`，只限制 longitudinal command；steering 原样保留。`info` 记录 proposed/applied command、介入原因、severity、风险及累计指标。评估报告中的 `safety_intervention_time_fraction` 必须与成功/碰撞率一起解释。

`observation=rgb_ego` 仅允许 MetaDrive 单学习 ego，结构为 `{"rgb": uint8[3,H,W], "motion": float32[20]}`。RGB 来自 MetaDrive `RGBCamera`；motion/navigation 为显式非图像信号。`ObservationEncoder('rgb_ego')` 使用 `RGBEncoder + motion MLP + fusion`，因此不是 camera-only policy。

`distill_actor()` 只训练 encoder + actor；critic/log_std 不更新。`scripts/distill_metadrive_rgb.py` 使用 frozen state teacher 在真实 MetaDrive 中执行 teacher action 并记录 RGB student observation。产物必须标记 `behavior_cloning_from_privileged_state_teacher`，不得当成独立 RGB RL 结果。


## v2.5 Safety policy distillation API

`src.safety.distillation.distill_safety_actor(model, states, targets, intervened, severity, ...)` 只更新 encoder + actor。transparent supervisor 样本保留 unit-weight 行为保持约束；真实介入样本按 severity 增权。critic 与 `log_std` 不更新，因此生成模型可用于 actor 部署/评估，但默认**不支持把该 checkpoint 当成完整 PPO optimizer resume 点**。

`scripts/distill_metadrive_safety.py` 在真实 MetaDrive 中让当前 actor 提议动作、`SafetySupervisor` 产生 applied longitudinal target，并支持按迭代收集 actor 自己的新 state distribution。正式 v2.5 模型只使用 release4 之前冻结的 collection/dev split；最终 release4 不参与候选选择。

评估新增 `safety_mean_longitudinal_correction`、`safety_peak_longitudinal_correction` 与 `safety_mean_severity`，用于量化组合系统对规则层的依赖，不能单独作为 safety certificate。`scripts/demo.py --config ...` 可在不改变 policy observation/agent schema 的前提下覆盖环境，例如对同一模型录制 raw 与 supervisor-on 系统。

## v2.7 Privileged expert 与 path suite

`src.experts.metadrive_idm.MetaDriveIDMTeacher` 是**训练期特权teacher**。它绑定真实 MetaDrive ego，并调用内置 `IDMPolicy` 生成 action target；teacher可能读取 simulator-internal navigation/lidar，因此不能在部署能力声明中算作 student observation。`scripts/distill_metadrive_expert.py` 的 `steering_only` 模式只更新 steering actor row，`balanced` 更新 encoder+actor；critic/log_std保持冻结。

`src.experts.distillation.distill_anchored_expert(...)` 同时最小化 expert target loss、frozen source action drift 和 frozen source feature drift。它用于研究 imitation representation drift，不提供安全保证。离线蒸馏每个epoch只产生一次全样本 permutation，确保minibatch不会因为重复shuffle而随机遗漏/重复样本。

`src.evaluation.path_suite.standard_path_suite(split, seeds_per_stage)` 提供 `curve_sparse/curve_moderate/mixed_sparse/mixed_moderate` 四个路径能力条件。`scripts/evaluate_safety_suite.py --suite path` 仍使用 fresh-process/commit-marker 机制，并可与 `--supervised` 组合；此组合只让纵向 SafetySupervisor工作，steering仍由policy输出。

v2.7新增 `expert_train`、`validation2`、`release6` seed domain。`select_safety_candidate.py --selection-split validation2` 只允许11000–11199；release6禁止进入选择函数。`reports/v2_7/FROZEN_MODEL.json` 记录selected model与selection report SHA，release6只能在该冻结记录之后用于诊断。

## v2.8 capability specialization APIs

### `src.evaluation.capability_suite.standard_capability_suite(split, seeds_per_stage=5)`

Returns five `SafetySuiteSpec` objects for curve tracking, ramp merge, bottleneck yield, intersection yield and dense progress. The split must come from the explicit seed registry in `src.scenarios.generator.SPLITS`.

### `src.evaluation.capability_selection.select_capability_candidate(...)`

Validation-only deterministic candidate ranking. It verifies that every model report uses the exact same stage/seed suite and rejects reports outside the allowed validation seed domain.

### `src.evaluation.capability_selection.capability_promotion_decision(...)`

Pre-release engineering gate. A candidate needs material success/catastrophic/timeout improvement while respecting collision, off-road and stage-local catastrophic-regression bounds. This is a registry gate, not a safety proof.

### `src.experts.componentwise.distill_bounded_components(...)`

Reference-bounded component-wise actor adaptation. The shared encoder is frozen; steering and longitudinal actor rows are optimized separately with soft margins to a frozen source policy. v2.8 retained this as a negative baseline because validation6 did not improve behavior.

### `scripts/distill_metadrive_capability.py`

Collects privileged MetaDrive IDM full-action labels and distills them into the deployable state actor. Setting both blend values to `1.0` produces pure-expert trajectories; the final v2.8 model used this mode. `IDMPolicy` is collection-only and is not instantiated at raw deployment.


## v2.9 route schema / transparent hybrid control

`EnvConfig.state_schema_version` supports `3` (legacy/default) and `4`. Schema4 keeps the state dimension unchanged but orders the four route preview points by forward path arc length; this avoids legacy ego-frame-x ordering failures on hairpin/backtracking geometry. Existing v2.8 checkpoints continue to use schema3 unless explicitly configured otherwise.

For MetaDrive state observations, `route_assist=True` enables transparent pure-pursuit steering. `route_assist_blend` is in `[0,1]`. The route layer never changes the longitudinal action. When SafetySupervisor is also enabled, ordering is:

```text
actor proposed action
    -> RouteAssist (steering only)
    -> SafetySupervisor (longitudinal only)
    -> MetaDrive
```

`info` exposes all three actions: `proposed_action`, `route_assisted_action`, `applied_action`, plus `route_assist_intervention`, `route_assist_target_steering`, `route_assist_steering_correction`, and the existing safety fields. Evaluation aggregates route and safety interventions separately. Hybrid metrics must not be labeled as raw-policy metrics.


## v3.0 adaptive shared control

`RouteAssist(mode="adaptive")` keeps the learned steering command and blends in pure-pursuit steering according to a policy-visible need score. `EnvConfig` exposes `route_assist_mode`, `route_assist_min_blend`, and curvature/heading/lane start/full thresholds. Runtime `info` reports `route_assist_blend`, `route_assist_need_score`, and `learned_steering_authority`.

Evaluation aggregates `route_assist_mean_blend`, `route_assist_mean_need_score`, and `learned_steering_authority_fraction`. These are shared-control authority metrics, not evidence that the underlying neural checkpoint learned the rule controller's skill.


## v3.1 learned route adapter

### `src.navigation.learned_route_adapter.LearnedRouteAdapter`

Neural residual steering module bound to a frozen base policy. Inputs are the flattened state observation and the base policy action. The module predicts a bounded steering residual and a learned blend; longitudinal action is passed through unchanged.

### `src.navigation.learned_route_adapter.LearnedRouteAdapterAgent`

Runtime wrapper around an existing `EndToEndAgent`. It verifies the adapter artifact's base-model SHA-256 before inference. No geometric RouteAssist is called by this wrapper.

### `scripts/train_learned_route_adapter.py`

Collects real MetaDrive online-DAgger data with adaptive RouteAssist as a training-only teacher and fits the residual adapter. Use only `adapter_train` for training.

### `scripts/evaluate_learned_route_suite.py`

Capability/safety/path suite evaluator with fresh-process stage execution, atomic completion reports, evaluation fingerprints and optional learned adapter / SafetySupervisor. Consumed release domains are for reproduction only, never parameter selection.
