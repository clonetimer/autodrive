import argparse
import json
import _bootstrap
from src.evaluation.benchmark import generalization_suite
from src.scenarios.generator import split_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reports = generalization_suite(args.model, split_seeds('test', args.episodes), args.output)
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
