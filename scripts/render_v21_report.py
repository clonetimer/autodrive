"""Render figures from measured v2.1 development results, not example numbers."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import _bootstrap


def render(root: Path, output: Path) -> None:
    summary = json.loads((root / 'aggregate_results.json').read_text())
    audit = json.loads((root / 'improvement_audit.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    names = ['cruise', 'dense', 'emergency']
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    for version, shift in [('2', -.19), ('3', .19)]:
        rates = [100 * summary[name][version]['success_rate'] for name in names]
        bars = ax.bar(x + shift, rates, width=.38, label=f'Reward v{version}')
        ax.bar_label(bars, fmt='%.1f%%', padding=3)
    ax.set_xticks(x, ['Cruise', 'Dense', 'Emergency braking'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Success rate (%)')
    ax.set_title('Paired development evaluation: gains and regression')
    ax.legend()
    fig.text(.5, .015, '3 training seeds; 40 shared scenarios per family. Not the final holdout.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output / 'reward_success_comparison.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for version in ['v2', 'v3']:
        rows = audit['reward'][version]['rows']
        ax.plot([row['actual_speed_mps'] for row in rows], [row['reward'] for row in rows],
                marker='o', label=f'Reward {version}')
    ax.set_xticks([12, 24])
    ax.set_xlabel('Actual speed (m/s); target = 12 m/s')
    ax.set_ylabel('Measured one-step reward')
    ax.set_title('Controlled regression: the overspeed incentive is removed')
    ax.legend()
    fig.text(.5, .015, 'Two measured states, joined for readability; not a full reward landscape or safety proof.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output / 'overspeed_reward_audit.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', default='reports/v2_1')
    parser.add_argument('--output', default='reports/v2_1/figures')
    args = parser.parse_args()
    render(Path(args.reports), Path(args.output))
    print(f'Measured-result figures written to {args.output}')
