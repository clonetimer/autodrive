"""Isolated CPU RGB-camera smoke test for MetaDrive.

Direct offscreen rendering may crash the interpreter when no display backend is
available. The parent therefore launches a disposable worker, using Xvfb when
DISPLAY is absent. A worker crash is recorded as evidence instead of taking down
the caller. This proves sensor/runtime availability only, not learned vision.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import _bootstrap
from src.utils.logger import save_json, runtime_manifest


def worker(report_path: Path, width: int, height: int, steps: int) -> int:
    report = {"status": "started", "steps": 0}
    env = None
    try:
        import numpy as np
        from metadrive import MetaDriveEnv
        from metadrive.component.sensors.rgb_camera import RGBCamera
        cfg = {
            "use_render": False,
            "image_observation": True,
            "image_on_cuda": False,
            "show_terrain": False,
            "show_interface": False,
            "show_logo": False,
            "show_fps": False,
            "start_seed": 0,
            "num_scenarios": 2,
            "horizon": max(steps + 2, 20),
            "map": "S",
            "traffic_density": 0.0,
            "log_level": 40,
            "decision_repeat": 5,
            "physics_world_step_size": 0.02,
            "stack_size": 1,
            "sensors": {"rgb_camera": (RGBCamera, width, height)},
            "vehicle_config": {"image_source": "rgb_camera"},
        }
        env = MetaDriveEnv(cfg)
        started = time.perf_counter()
        observation, _ = env.reset(seed=0)
        image = np.asarray(observation["image"])
        first = {"shape": list(image.shape), "dtype": str(image.dtype),
                 "minimum": float(image.min()), "maximum": float(image.max()),
                 "mean": float(image.mean())}
        step_seconds = []
        for _ in range(steps):
            tick = time.perf_counter()
            observation, _, terminated, truncated, _ = env.step([0.0, 0.2])
            step_seconds.append(time.perf_counter() - tick)
            report["steps"] += 1
            if terminated or truncated:
                break
        image = np.asarray(observation["image"])
        report.update(status="passed", first_frame=first,
                      last_frame_mean=float(image.mean()),
                      reset_plus_steps_seconds=time.perf_counter() - started,
                      mean_step_seconds=float(np.mean(step_seconds)) if step_seconds else None,
                      renderer="MetaDrive RGBCamera through Panda3D, CPU image_on_cuda=False")
        return 0
    except Exception as exc:
        report.update(status="failed", exception=type(exc).__name__, message=str(exc))
        return 2
    finally:
        if env is not None:
            env.close()
        save_json(report, report_path)


def parent(output: Path, width: int, height: int, steps: int, timeout: float, use_existing_display: bool = False) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="metadrive_rgb_") as temp:
        worker_report = Path(temp) / "worker.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--worker-report", str(worker_report), "--width", str(width),
                   "--height", str(height), "--steps", str(steps)]
        display_mode = "existing_DISPLAY"
        xvfb = shutil.which("xvfb-run")
        if xvfb and not use_existing_display:
            command = [xvfb, "-a", "-s", "-screen 0 1024x768x24", *command]
            display_mode = "xvfb-run"
        elif not os.environ.get("DISPLAY"):
            display_mode = "none"
        started = time.perf_counter()
        try:
            process = subprocess.run(command, cwd=_bootstrap.ROOT, capture_output=True,
                                     text=True, timeout=timeout, check=False)
            rc, timed_out = process.returncode, False
            stdout, stderr = process.stdout, process.stderr
        except subprocess.TimeoutExpired as exc:
            rc, timed_out = None, True
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        child = json.loads(worker_report.read_text()) if worker_report.is_file() else None
        result = {"runtime": runtime_manifest(), "status": "passed" if child and child.get("status") == "passed" and rc == 0 else "blocked_or_failed",
                  "display_mode": display_mode, "command": command, "return_code": rc,
                  "timed_out": timed_out, "elapsed_seconds": time.perf_counter() - started,
                  "worker": child, "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-3000:]}
        save_json(result, output)
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default=str(_bootstrap.ROOT / "reports/metadrive_camera_smoke.json"))
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--use-existing-display", action="store_true", help="Skip Xvfb even if available")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--worker-report")
    args = p.parse_args()
    if args.worker:
        if not args.worker_report:
            p.error("--worker-report is required in worker mode")
        raise SystemExit(worker(Path(args.worker_report), args.width, args.height, args.steps))
    result = parent(Path(args.output), args.width, args.height, args.steps, args.timeout, args.use_existing_display)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
