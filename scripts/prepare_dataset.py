import argparse
import _bootstrap
from src.datasets.scenario import csv_to_scenario


def main():
    parser = argparse.ArgumentParser(description='Convert documented CSV trajectories to safe portable JSON (not raw Waymo/nuScenes).')
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--provenance', required=True)
    args = parser.parse_args()
    result = csv_to_scenario(args.csv, args.output, args.provenance)
    print(f"Converted {len(result['tracks'])} tracks")


if __name__ == '__main__':
    main()
