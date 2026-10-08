"""Run on a compatible networked machine; create auditable offline wheels.

Use the intended Python/OS/architecture, or explicitly supply cross-target pip
arguments. Cross-target mode only accepts published binary wheels; source-only
packages must be built on a matching machine (do not rename wheel tags).
"""
import argparse
import subprocess
import sys
from pathlib import Path
import _bootstrap
from src.utils.installation import build_wheel_manifest
from src.utils.logger import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requirements', default='requirements-metadrive.txt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--platform')
    parser.add_argument('--python-version')
    parser.add_argument('--abi')
    parser.add_argument('--implementation', default='cp')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cross = bool(args.platform or args.python_version or args.abi)
    if cross and not all([args.platform, args.python_version, args.abi]):
        parser.error('Cross-target download requires --platform, --python-version and --abi together')
    command = [sys.executable, '-m', 'pip', 'download' if cross else 'wheel',
               '--dest' if cross else '--wheel-dir', str(output), '-r', str(Path(args.requirements).resolve()), '-r', str(_bootstrap.ROOT / 'requirements-build.txt')]
    if cross:
        command += ['--only-binary=:all:', '--platform', args.platform, '--python-version', args.python_version,
                    '--abi', args.abi, '--implementation', args.implementation]
    save_json({'command': command, 'cross_target': cross}, output / 'build_request.json')
    subprocess.run(command, check=True)
    manifest = build_wheel_manifest(output, output / 'manifest.json')
    print(f"Recorded {len(manifest['wheels'])} wheels; run target-side bootstrap and pip check to verify closure.")


if __name__ == '__main__':
    main()
