"""Reliable CPU MetaDrive training via bounded fresh-process segments.

Each segment invokes the project's real trainer in a child process. Checkpoint
weights, optimizer and counters resume; the simulator itself restarts at a new
episode boundary. A stalled simulator process is killed after a bounded timeout
and retried from the last accepted model. No native fallback is permitted.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import _bootstrap
from src.utils.config import load_config
from src.utils.logger import save_json, sha256
from src.utils.segmented import segment_targets, completed_training_artifact


def terminate(process: subprocess.Popen, grace: float = 3.0) -> str:
    """Terminate the whole worker process group; Panda3D/Xvfb may own children."""
    if process.poll() is not None:
        return "already_exited"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_exited"
    try:
        process.wait(timeout=grace)
        return "terminated_group"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace)
            return "killed_group"
        except subprocess.TimeoutExpired:
            return "kill_sent_process_not_reaped"


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    config = load_config(config_path, {"total_steps": args.total_steps})
    if config.env.backend != "metadrive":
        raise ValueError("This reliability runner is intentionally MetaDrive-only")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    segments_root = output / "segments"
    segments_root.mkdir()
    manifest = {
        "status": "started",
        "backend": "metadrive",
        "total_steps": args.total_steps,
        "segment_steps": args.segment_steps,
        "segment_timeout_seconds": args.timeout,
        "max_retries": args.max_retries,
        "resume_semantics": "model/optimizer/counter continue; each process begins a fresh simulator episode",
        "segments": [],
    }
    resume = Path(args.resume).resolve() if args.resume else None
    start = 0
    if resume:
        import torch
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        start = int(payload.get("steps", 0))
    targets = segment_targets(start, args.total_steps, args.segment_steps)
    try:
        for index, target in enumerate(targets, 1):
            accepted = None
            for attempt in range(1, args.max_retries + 2):
                segment_dir = segments_root / f"segment_{index:02d}_target_{target}_attempt_{attempt}"
                segment_dir.mkdir(parents=True, exist_ok=False)
                log_path = segment_dir / "process.log"
                command = [sys.executable, str(_bootstrap.ROOT / "scripts/train.py"),
                           "--config", str(config_path), "--output", str(segment_dir),
                           "--steps", str(target)]
                if resume:
                    command += ["--resume", str(resume)]
                display_mode = "none_required"
                if config.env.observation == "rgb_ego":
                    xvfb = shutil.which("xvfb-run")
                    if xvfb is not None:
                        # Always isolate camera workers from a stale/foreign DISPLAY.
                        command = [xvfb, "-a", "-s", "-screen 0 1024x768x24", *command]
                        display_mode = "xvfb-run"
                    elif os.environ.get("DISPLAY"):
                        display_mode = "existing_DISPLAY"
                    else:
                        raise RuntimeError("rgb_ego segmented training requires DISPLAY or xvfb-run")
                started = time.monotonic()
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=_bootstrap.ROOT, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                        start_new_session=True)
                    termination = "not_needed"
                    timed_out = False
                    while process.poll() is None:
                        artifact = completed_training_artifact(segment_dir, target)
                        if artifact is not None:
                            # Result/model are atomically complete enough to accept; allow
                            # normal interpreter shutdown briefly, then prevent a lingering
                            # Panda3D process from blocking the workflow.
                            try:
                                process.wait(timeout=args.post_result_grace)
                            except subprocess.TimeoutExpired:
                                termination = terminate(process)
                            break
                        if time.monotonic() - started > args.timeout:
                            timed_out = True
                            termination = terminate(process)
                            break
                        time.sleep(0.25)
                    rc = process.poll()
                artifact = completed_training_artifact(segment_dir, target)
                row = {"index": index, "target_steps": target, "attempt": attempt,
                       "command": command, "return_code": rc, "timed_out": timed_out,
                       "termination": termination, "elapsed_seconds": time.monotonic() - started,
                       "artifact_complete": artifact is not None,
                       "directory": str(segment_dir.relative_to(output)), "display_mode": display_mode}
                if artifact is not None:
                    row["model_sha256"] = sha256(segment_dir / "model.pt")
                    accepted = segment_dir
                    manifest["segments"].append(row)
                    break
                manifest["segments"].append(row)
            if accepted is None:
                raise RuntimeError(f"Segment target {target} failed after retries")
            resume = accepted / "model.pt"
        final_model = output / "model.pt"
        shutil.copy2(resume, final_model)
        final_segment = completed_training_artifact(resume.parent, args.total_steps)
        manifest.update(status="segmented_training_completed_not_safety_claim",
                        final_model_sha256=sha256(final_model), final_world_steps=args.total_steps,
                        final_segment_result=final_segment)
        save_json(manifest, output / "result.json")
        return manifest
    except Exception as exc:
        manifest.update(status="blocked_or_failed", exception=type(exc).__name__, message=str(exc))
        save_json(manifest, output / "result.json")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--total-steps", type=int, required=True)
    p.add_argument("--segment-steps", type=int, default=8192)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--post-result-grace", type=float, default=3.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--resume")
    args = p.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
