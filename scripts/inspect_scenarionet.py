"""Audit existing numerical ScenarioNet examples and optionally plot recorded data."""
import argparse
from pathlib import Path
import _bootstrap
from src.datasets.scenarionet_inspect import numerical_load, inspect_directory
from src.utils.logger import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--figures')
    args = p.parse_args()
    report = {'real_metadrive_steps': 0}
    try:
        report = inspect_directory(args.directory)
        if args.figures:
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            dest = Path(args.figures)
            dest.mkdir(parents=True, exist_ok=True)
            report['recorded_data_figures'] = []
            for row in report['scenarios']:
                data = numerical_load(Path(args.directory) / row['file'])
                fig, ax = plt.subplots(figsize=(8, 6))
                ego = data['tracks'][row['recorded_ego_id']]['state']
                valid = np.asarray(ego['valid']).reshape(-1).astype(bool)
                origin = np.asarray(ego['position'])[valid][0, :2]
                drawn = 0
                for feature in data['map_features'].values():
                    xy = np.asarray(feature.get('polyline', []))
                    if xy.ndim == 2 and len(xy) and np.linalg.norm(xy[:, :2] - origin, axis=1).min() < 80:
                        ax.plot(xy[:, 0] - origin[0], xy[:, 1] - origin[1], linewidth=.5, alpha=.35,
                                label='Recorded map polylines' if drawn == 0 else None)
                        drawn += 1
                xy = np.asarray(ego['position'])[valid]
                ax.plot(xy[:, 0] - origin[0], xy[:, 1] - origin[1], linewidth=3, label='Recorded ego trajectory')
                ax.set(xlim=(-80, 80), ylim=(-60, 60), xlabel='x relative to initial ego (m)',
                       ylabel='y relative to initial ego (m)',
                       title=f'Waymo example {row["scenario_id"]}\nRecorded data only — not a simulation rollout')
                ax.set_aspect('equal'); ax.legend(loc='best'); fig.tight_layout()
                path = dest / (row['scenario_id'] + '.png')
                fig.savefig(path, dpi=140); plt.close(fig)
                report['recorded_data_figures'].append(str(path))
        code = 0
    except Exception as exc:
        report.update(status='failed', exception=type(exc).__name__, message=str(exc))
        code = 2
    save_json(report, args.output)
    print(report['status'], args.output)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
