"""JSON/CSV/Markdown/HTML from the same recorded evaluation object."""
from __future__ import annotations
from pathlib import Path
import csv
import html
from src.utils.logger import save_json


def export_report(report: dict, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(report, output / 'evaluation.json')
    rows = report['episodes']
    if rows:
        with (output / 'episodes.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [f"# {report['label']}", '', f"Backend: `{report['backend']}`; input: `{report['observation']}`.",
             f"Scenario seeds: {report['seeds']}", '', '| Metric | Value |', '|---|---|']
    for key, value in report['metrics'].items():
        display = f'{value:.6f}' if isinstance(value, float) else str(value)
        lines.append(f'| {key} | {display} |')
    lines += ['', 'All episodes, including crashes and timeouts, are present in episodes.csv.',
              'These are short-budget simulation measurements, not deployment safety evidence.']
    (output / 'evaluation.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    table = ''.join(f'<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>' for key, value in report['metrics'].items())
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Driving evaluation</title>'
                '<style>body{font:16px sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}th,td{padding:.6em;text-align:left;border-bottom:1px solid #aaa}table{border-collapse:collapse}</style>'
                f"<h1>{html.escape(report['label'])}</h1><p>Backend: {html.escape(report['backend'])}</p><table>{table}</table>"
                '<p>Empirical simulation result; failures are retained. Not a safety certification.</p></html>')
    (output / 'evaluation.html').write_text(document, encoding='utf-8')
