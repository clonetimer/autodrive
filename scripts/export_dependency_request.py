"""Export exact target interpreter/tags/installed metadata for an additive bundle."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import zipfile
import _bootstrap
from packaging.markers import default_environment
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name
from src.utils.logger import save_json, sha256

# Audited against supplied MetaDrive 0.4.3 setup.py, plus project restrictions.
ROOTS = ['requests', 'matplotlib', 'pygame>=2.5', 'yapf', 'tqdm', 'progressbar==2.5',
         'pillow', 'lxml', 'psutil', 'shapely', 'filelock', 'Pygments', 'mediapy',
         'panda3d>=1.10.15,<1.11', 'panda3d-gltf>=0.13,<1', 'gymnasium>=0.29,<1.3',
         'numpy>=1.26,<3', 'opencv-python>=4.8', 'setuptools>=68', 'wheel',
         'stable-baselines3==2.7.1', 'tensorboard>=2.16,<3', 'pettingzoo>=1.24,<2']


def export(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    installed = {}
    for dist in metadata.distributions():
        name = canonicalize_name(dist.metadata['Name'])
        installed[name] = {'name': name, 'version': dist.version,
                           'requires_dist': dist.requires or [],
                           'requires_python': dist.metadata.get('Requires-Python', '')}
    tags = list(sys_tags())
    # Ordered manylinux compatibility tags from the TARGET, not the downloader OS.
    platforms = list(dict.fromkeys(t.platform for t in tags if t.platform != 'any'))
    target = {'schema': 1, 'python_version': f'{sys.version_info.major}.{sys.version_info.minor}',
              'implementation': 'cp', 'abis': list(dict.fromkeys([tags[0].abi, 'abi3', 'none'])),
              'platforms': platforms, 'compatible_tags': list(map(str, tags)),
              'marker_environment': default_environment(), 'installed': installed,
              'root_requirements': ROOTS, 'platform': platform.platform(),
              'note': 'Delta bundle: reuse this exact base, do not replace it with a clean unrelated Python installation.'}
    target['fingerprint'] = hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()
    save_json(target, output / 'target.json')
    shutil.copyfile(_bootstrap.ROOT / 'tools/collect_metadrive_wheels.py', output / 'collect_metadrive_wheels.py')
    (output / 'README_先读我.md').write_text('''# MetaDrive 依赖补齐工具（不是依赖包本身）\n\n当前目标：Linux x86_64，CPython 3.13。脚本读取 target.json，不会误下载 Windows/macOS 原生包。\n\n在联网电脑安装 Python 与 pip 后，解压并在此目录运行：\n\n```bash\npython collect_metadrive_wheels.py --output metadrive_wheelhouse.zip\n```\n\n上传生成的 **metadrive_wheelhouse.zip**。不需要再次上传 assets.zip 或源码。\n脚本默认仅下载目标沙盒缺失项及其传递依赖；不会重下现有 CPU PyTorch。\n不在联网电脑安装 MetaDrive，不需要 GPU 或管理员权限。progressbar 仅有源码发行，脚本会在临时构建环境生成真实纯 Python wheel，并校验其平台标签。\n\n离线预览：`python collect_metadrive_wheels.py --plan-only`。\n若失败，上传生成的 `.failure.json`（不是空目录）。脚本不会生成虚假的成功包。\n\n该下载路径已做离线单元测试；此沙盒 DNS 不通，所以尚未端到端下载验收。\nmanifest 为哈希校验而非数字签名；目标端还必须执行 pip check、真实导入与物理 step。\n''', encoding='utf-8')
    archive_path = output.with_suffix('.zip')
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.iterdir()):
            if p.is_file():
                archive.write(p, output.name + '/' + p.name)
    return {'directory': str(output), 'archive': str(archive_path), 'sha256': sha256(archive_path),
            'target_fingerprint': target['fingerprint'], 'roots': ROOTS}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=str(_bootstrap.ROOT / 'reports/v2_2/dependency_request'))
    print(json.dumps(export(p.parse_args().output), indent=2, ensure_ascii=False))
