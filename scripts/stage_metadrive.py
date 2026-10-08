"""Verify and stage user-supplied MetaDrive source + assets offline (no pip)."""
import argparse
from pathlib import Path
import _bootstrap
from src.utils.metadrive_assets import stage_assets, audit_upload
from src.utils.logger import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-zip', required=True)
    p.add_argument('--assets-zip', required=True)
    p.add_argument('--installation-zip')
    p.add_argument('--workspace', default=str(_bootstrap.ROOT / 'external'))
    p.add_argument('--report', default=str(_bootstrap.ROOT / 'reports/metadrive_staging.json'))
    args = p.parse_args()
    report = {'real_steps_executed': 0, 'pip_executed': False}
    code = 0
    try:
        report['uploads'] = [audit_upload(path) for path in [args.source_zip, args.assets_zip, args.installation_zip] if path]
        report.update(stage_assets(args.source_zip, args.assets_zip, args.workspace))
    except Exception as exc:
        report.update(status='failed', exception=type(exc).__name__, message=str(exc))
        code = 2
    save_json(report, args.report)
    print(report.get('status'), args.report)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
