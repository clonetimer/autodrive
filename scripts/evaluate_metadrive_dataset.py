"""Closed-loop evaluation on converted ScenarioNet/Waymo directories in MetaDrive.

The ego is controlled by the frozen learned policy. Background behavior follows
MetaDrive ScenarioEnv configuration; this is not claimed as fully reactive
counterfactual traffic unless reactive_traffic is explicitly implemented.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import _bootstrap
from src.end_to_end.agent import EndToEndAgent
from src.evaluation.evaluator import evaluate
from src.utils.logger import save_json


def scenario_count(directory: Path) -> int:
    files = sorted(p for p in directory.glob("*.pkl") if p.name != "dataset_summary.pkl")
    if not files:
        raise FileNotFoundError("No converted scenario .pkl files found")
    return len(files)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scenarios", type=int, default=0, help="0 means every converted scenario")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--trajectories", action="store_true")
    args = p.parse_args()
    dataset = Path(args.dataset).resolve()
    available = scenario_count(dataset)
    count = available if args.scenarios == 0 else min(args.scenarios, available)
    if count <= 0:
        raise ValueError("scenarios must be nonnegative and at least one scenario must be selected")
    agent = EndToEndAgent(args.model)
    if agent.config.env.num_agents != 1:
        raise ValueError("ScenarioEnv dataset evaluation currently supports single-agent checkpoints")
    config = replace(agent.config.env, backend="metadrive", dataset=str(dataset), num_agents=1,
                     max_steps=args.max_steps)
    report = evaluate(lambda name, seed: agent, config, list(range(count)), args.output,
                      label=f"dataset:{dataset.name}", save_trajectories=args.trajectories)
    metadata = {"dataset": str(dataset), "available_scenarios": available, "evaluated_scenarios": count,
                "model": str(Path(args.model).resolve()), "mode": "closed_loop_ego_on_MetaDrive_ScenarioEnv",
                "background_note": "ScenarioEnv background is not claimed as fully reactive counterfactual traffic",
                "metrics": report["metrics"]}
    save_json(metadata, Path(args.output) / "dataset_evaluation.json")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
