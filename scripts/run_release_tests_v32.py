"""Run the v3.2 release test suite while isolating MetaDrive/Panda3D teardown.

MetaDrive/Panda3D can finish pytest's test phase and print a passing summary yet
remain alive in interpreter teardown.  The release runner treats a *complete
pytest summary with no failures/errors* as the functional result, gives normal
teardown a grace period, then terminates the process group if required.  This
behavior is recorded explicitly as ``teardown_forced`` rather than silently
reported as a normal process exit.

Non-MetaDrive tests are run in one process for speed.  Every real MetaDrive test
node is run in its own process so engine lifetime and global state cannot leak
between integration tests.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

PASS_RE = re.compile(r"(?P<n>\d+) passed")
SKIP_RE = re.compile(r"(?P<n>\d+) skipped")
DESEL_RE = re.compile(r"(?P<n>\d+) deselected")
FAIL_RE = re.compile(r"(?P<n>\d+) failed")
ERROR_RE = re.compile(r"(?P<n>\d+) errors?")


def _one(rx: re.Pattern[str], text: str) -> int:
    matches = list(rx.finditer(text))
    return int(matches[-1].group("n")) if matches else 0


def _counts(text: str) -> dict[str, int]:
    return {
        "passed": _one(PASS_RE, text),
        "skipped": _one(SKIP_RE, text),
        "deselected": _one(DESEL_RE, text),
        "failed": _one(FAIL_RE, text),
        "errors": _one(ERROR_RE, text),
    }


def _summary_complete(text: str) -> bool:
    counts = _counts(text)
    # A real summary has at least one terminal category.  We only accept one
    # that contains no failed/error tests.
    return (counts["passed"] + counts["skipped"] + counts["failed"] + counts["errors"] > 0
            and counts["failed"] == 0 and counts["errors"] == 0
            and (" passed" in text or " skipped" in text))


def _terminate_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass


def _run(command: list[str], cwd: Path, timeout: float, grace_after_summary: float = 2.0) -> dict:
    """Run a pytest command and distinguish test completion from teardown hang."""
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False, dir=cwd) as log:
        log_path = Path(log.name)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TERM": "xterm", "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )

    start = time.monotonic()
    summary_seen_at: float | None = None
    timed_out = False
    teardown_forced = False
    while process.poll() is None:
        elapsed = time.monotonic() - start
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if _summary_complete(text):
            if summary_seen_at is None:
                summary_seen_at = time.monotonic()
            elif time.monotonic() - summary_seen_at >= grace_after_summary:
                teardown_forced = True
                _terminate_group(process)
                break
        if elapsed >= timeout:
            timed_out = True
            _terminate_group(process)
            break
        time.sleep(0.25)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    log_path.unlink(missing_ok=True)
    counts = _counts(text)
    summary_ok = _summary_complete(text)
    return {
        "process_return_code": process.returncode,
        "timed_out": timed_out,
        "teardown_forced": teardown_forced,
        "summary_complete": summary_ok,
        "counts": counts,
        "stdout_tail": text[-5000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", default="reports/v3_2/release_tests.json")
    parser.add_argument("--non-meta-timeout", type=float, default=120.0)
    parser.add_argument("--meta-timeout", type=float, default=90.0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    py = Path(sys.executable)
    records: list[dict] = []

    print("[non_meta] full partition", flush=True)
    non_command = [str(py), "-m", "pytest", "-q", "-m", "not metadrive"]
    non = _run(non_command, root, args.non_meta_timeout)
    non.update({"partition": "non_metadrive", "target": "-m not metadrive", "command": non_command})
    records.append(non)
    if not non["summary_complete"] or non["timed_out"]:
        raise SystemExit(json.dumps({"failed": non}, indent=2))

    collect = subprocess.run(
        [str(py), "-m", "pytest", "--collect-only", "-q", "-m", "metadrive"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    nodes = [
        line.strip() for line in collect.stdout.splitlines()
        if line.strip().startswith("tests/") and "::" in line
    ]
    for node in nodes:
        print(f"[metadrive] {node}", flush=True)
        command = [str(py), "-m", "pytest", "-q", node]
        record = _run(command, root, args.meta_timeout)
        record.update({"partition": "metadrive", "target": node, "command": command})
        records.append(record)
        if not record["summary_complete"] or record["timed_out"]:
            raise SystemExit(json.dumps({"failed": record}, indent=2))

    passed = sum(r["counts"]["passed"] for r in records)
    skipped = sum(r["counts"]["skipped"] for r in records)
    result = {
        "passed": passed,
        "skipped": skipped,
        "failed": 0,
        "non_metadrive": {
            "passed": non["counts"]["passed"],
            "skipped": non["counts"]["skipped"],
            "teardown_forced": non["teardown_forced"],
        },
        "metadrive": {
            "nodes": len(nodes),
            "passed": sum(r["counts"]["passed"] for r in records if r["partition"] == "metadrive"),
            "skipped": sum(r["counts"]["skipped"] for r in records if r["partition"] == "metadrive"),
            "teardown_forced_nodes": sum(bool(r["teardown_forced"]) for r in records if r["partition"] == "metadrive"),
        },
        "isolation": "all non-MetaDrive tests in one pytest process; every real-MetaDrive node in its own process",
        "teardown_policy": "A complete passing pytest summary is accepted; a process still alive after a 2s grace is terminated and recorded as teardown_forced.",
        "records": records,
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
