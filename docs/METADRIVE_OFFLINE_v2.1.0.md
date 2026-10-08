# 真实 MetaDrive：可诊断、可校验的安装路径

## 本轮证据

用户源码 version=0.4.3，SHA-256 `4a031149cad2fe3d25570620e9a868c48295065c2b6b4951f6c3e765383475c7`，不含匹配 assets。
`reports/v2_1/network_probe.json` 记录 pypi.org、files.pythonhosted.org、github.com DNS 失败；网络失败发生在依赖解析之前，不能解释为包不存在或确定 Python 版本不兼容。
真实 doctor 尝试失败、real_steps_executed=0。没有 GPU 不是这些阻塞的原因。

## 1. 只检查，不污染全局环境

```bash
python scripts/bootstrap_metadrive.py --source-zip /path/to/metadrive.zip \
  --preflight-only --report reports/installation_preflight.json
```

source版本通过 AST/literal 读取，不执行 ZIP 中 setup.py。网络是受限时长探针，DNS 失败不重复 pip 重试。

## 2. 在可联网、与目标环境匹配的机器上准备依赖

```bash
python scripts/build_wheelhouse.py \
  --requirements requirements-metadrive.txt --output /path/to/wheelhouse
```

wheelhouse 包含解析出的依赖和构建工具 wheel，附 manifest.json。可显式传 --platform、--python-version、--abi 做 binary-only 跨目标下载；源代码构建包应在匹配的 OS/架构/解释器上构建，不改名伪造 wheel 标签。`requirements-metadrive.txt` 包含项目与实仿真依赖；原始数据处理或渲染还可能要求其他系统动态库。

再从可信官方渠道取得与源码版本相符的 assets.zip。清单应从可信通信渠道获得，哈希不是第三方签名。原源码和 assets 不打入本项目交付 ZIP。

## 3. 在目标机创建隔离环境并完成离线安装

```bash
python -m venv .venv-metadrive
source .venv-metadrive/bin/activate
python -m pip install --no-index --find-links /path/to/wheelhouse \
  -r requirements.txt -r requirements-build.txt
python scripts/bootstrap_metadrive.py --source-zip /path/to/metadrive.zip \
  --wheelhouse /path/to/wheelhouse \
  --wheelhouse-manifest /path/to/wheelhouse/manifest.json \
  --assets-zip /path/to/assets.zip \
  --workspace external --report reports/installation.json
```

可加 `--expected-source-sha256`、`--expected-assets-sha256`。门禁依次检查源码、资源、wheel清单/内容/标签、隔离环境、pip解析/构建、assets放置、pip check、真实doctor。源码editable安装会执行可信源码的构建过程。实际安装仍可能暴露系统库/版本问题，不把预检通过称为安装成功。完整离线包及安装成功本轮没有完成验证。

## 4. 先状态无渲染，再训练，最后图像

```bash
python scripts/doctor.py --smoke --output reports/my_doctor.json
python -m pytest -q -m metadrive
python scripts/train_sb3_reference.py --help
python scripts/train.py --config configs/metadrive.yaml --output experiments/real_metadrive
```

推荐门槛仍为100种子累计至少10000真实step，再做短PPO训练/冷加载/闭环评估；这些门槛**尚未达到**。SB3脚本需要真实 Gymnasium/SB3，默认原生状态作为参考；MetaDrive为另一backend选项，不会静默替换缺失依赖。参考算法网络/动作参数化与自研实现不完全相同，不能仅凭结果作严格算法公平排名。

无窗口状态仿真与 offscreen RGB 渲染是两个验收项目。关闭渲染可降低资源需求，但不保证所有CPU机器都能软件渲染，更不代表原始视觉训练已经完成。
