# v2.2.0：MetaDrive 资源落盘、离线依赖补齐与 CPU 验收

## 本轮的实际结论

用户提供的源码与 assets 均为 **0.4.3**。`assets.zip` 包含 152 个文件（另外有目录项），全部通过 ZIP CRC、SHA-256、版本及 7 处 glTF 外部文件引用检查，并真实部署到 `external/metadrive/metadrive/assets/`。重复执行再次校验成功，没有只检查一个 version.txt 就宣布完整。

已经创建 `external/runtime` CPU 依赖覆盖环境，通过 `.pth` 显式复用原 CPU PyTorch 基础库；pip 写入新环境，不改原始基础库。在此环境内真实完成 `metadrive-simulator==0.4.3` 源码的 editable 安装（`--no-index --no-deps --no-build-isolation`）。**这只是源码注册与安装，不等于引擎能够导入、运行或训练。**

`MetaDrive_0.4.3_install.zip` 仅有 4 个文件：MD/PDF/HTML 文档及联网 shell 脚本，**0 个 wheel**。当前仍缺 `gymnasium`、`panda3d`、`pygame`、`progressbar`、`panda3d-gltf`、`mediapy`、`yapf` 七个核心依赖；另外 SB3、TensorBoard、PettingZoo 和 wheel 构建工具也未安装。真实引擎和正式训练执行步数仍为 **0**。

证据：`reports/v2_2/assets_staging.json`、`assets_final_check.json`、`metadrive_doctor.json`、`dependency_closure.json`；`logs/v2_2/source_install.log`、`runtime_pip_check.log`、`dependency_install_attempt.log`。再次联网检查仍为 DNS 失败；PyPI 的“找不到版本”结果不能据此解释成该版本不存在。

## A. 用户只需在联网电脑运行依赖收集工具

解压独立的 `metadrive_dependency_request.zip`，在该目录运行：

```bash
python collect_metadrive_wheels.py --output metadrive_wheelhouse.zip
```

上传生成的 `metadrive_wheelhouse.zip`，而不是再次上传下载器脚本、安装说明或 assets。收集器按 `target.json` 中 **Linux x86_64 / CPython 3.13** 的标签选择文件，而非联网电脑的 Windows/macOS 标签。Python 3.10+ 与 pip 即可启动，packaging 缺失时使用 pip 已附带的实现。

它复用目标沙盒已经存在的 CPU torch/numpy 等精确版本，仅下载缺失项及传递依赖；不在联网电脑安装 MetaDrive，也不需要 GPU。依赖环境标记在目标环境下求值，不误装 Windows 专用依赖。遇到冲突会停止并生成 `.failure.json`，不会把不完整目录压成“成功包”。

`progressbar==2.5` 的官方 PyPI 发行仅提供源码包，因此脚本调用 pip 在临时构建环境生成真正的纯 Python wheel，并验证 `none-any` 标签；不是把 `.tar.gz` 改后缀。该网络构建/下载链路未在当前断网沙盒端到端通过，已有离线依赖图、冲突、目标 ABI、环境标记、纯元数据、退出路径的单元测试。

```bash
# 不联网的预览，不会生成虚假的依赖包
python collect_metadrive_wheels.py --plan-only
```

生成包内：实际 `.whl`、`manifest.json`、`resolution.json`、`target.json`、`download_commands.json` 和版本锁定文件。manifest 是哈希一致性证明，不是供应链签名。目标基础环境改变后，delta 包会拒绝直接复用，应重新导出 target。

## B. 离线资源部署（已实测）

```bash
python scripts/stage_metadrive.py \
  --source-zip ../metadrive.zip --assets-zip ../assets.zip \
  --workspace external --report reports/my_assets.json

python scripts/create_metadrive_runtime.py \
  --output external/runtime --report reports/my_runtime.json
```

上述创建运行环境的命令不会覆盖已有目录。资源部署则在相同输入时重复校验；发现源码或资源被修改时拒绝覆盖。该覆盖环境继承了基础库，不是完全独立的全新 Docker 环境；基础库中 moviepy/Pillow 的无关版本冲突仍如实记录。

## C. 取得真实 wheels 后的目标端安装

将 wheelhouse ZIP 安全解压到 `external/wheelhouse`（压缩包自身有 wheelhouse/ 根目录）：

```bash
python -c "from src.datasets.download import safe_extract_zip; safe_extract_zip('../metadrive_wheelhouse.zip','external')"

external/runtime/bin/python scripts/bootstrap_metadrive.py \
  --source-zip ../metadrive.zip --assets-zip ../assets.zip \
  --workspace external --wheelhouse external/wheelhouse \
  --allow-unrelated-base-conflicts \
  --report reports/my_metadrive_installation.json
```

`--allow-unrelated-base-conflicts` 不是无条件跳过依赖检查：模拟器/核心项目的实际依赖图必须通过，目标的精确基础版本也必须匹配；只有继承来的无关包冲突可显式保留警告。全局 `pip check` 的失败日志不会被改为通过。不使用该开关时任何全局冲突都会阻止后续运行。

安装后还须实际执行动态库导入、真实 reset/step 和训练，wheel 标签正确并不代表 Panda3D 动态链接已验证。

## D. 官方观测与官方奖励的 CPU 参考训练入口

```bash
external/runtime/bin/python scripts/train_metadrive_official.py \
  --output experiments/my_metadrive_official \
  --steps 32768 --train-start 0 --train-count 100 \
  --eval-start 1000 --episodes 20 --seed 7
```

先保持官方观测和官方奖励，避免同时调试自定义适配器与算法。程序创建真实 MetaDrive 和真实 SB3，使用 CPU、检查 Gymnasium 接口、保存 checkpoint/CSV/TensorBoard；关闭训练引擎后冷加载最终模型并创建评估引擎，避免同进程同时存在两个 Panda3D 引擎。训练场景与评估场景不重叠；失败和零执行步数记录在 result.json，绝不切换到 native。

本轮实际启动该命令，被缺失依赖门禁阻止，见 `experiments/v2_2/metadrive_official/result.json`；没有生成训练权重。

用户安装脚本中旧的四元组 step、`env_num=1`、`headless` 参数不在这个新入口使用。真正的 Gymnasium 流程是 `obs, info = reset()` 和 `obs, reward, terminated, truncated, info = step(action)`。这里以提供的 0.4.3 源码 API 为准，而不是盲目执行上传脚本中的安装后自检。

## E. 已提供的真实数据样例

assets 已包含 3 个转换后的 Waymo 示例，每个 91 帧，共 17,791 个有效 track-frame 状态。本轮真实读取、验证时间戳/有效掩码/轨迹/地图形状，交叉检查 dataset_summary，并绘制了录制地图与自车轨迹。pickle reader 只允许预期的数值重建操作，拒绝外部执行器和 object dtype，但它不是通用恶意 pickle 沙箱。

```bash
python scripts/inspect_scenarionet.py \
  --directory external/metadrive/metadrive/assets/waymo \
  --output reports/my_waymo_audit.json --figures reports/my_waymo_figures
```

这是现有真实数据的审计，不是新下载的完整 Waymo 数据，不是 MetaDrive 仿真回放，也不是可响应 RL 的背景交通。原数据与第三方 assets 不重新打包进本项目发布 ZIP；部署脚本可使用用户已上传的原文件重建。

## F. 本轮适配器实质修复

移除 ScenarioEnv 不接受的程序道路专属键（包括 `out_of_road_done`），改用该环境自身的碰撞/出路终止约定；记录 `crash_human`；转向由车辆实际 `max_steering`（度）转换到弧度，移除旧常数 0.45；补充横向加速度、横向 jerk 和转向率输出。纯配置/单位测试通过，但真实适配器执行仍需上述依赖解除后验收。

## 官方参考（2026-09-07 在线核对）

- MetaDrive 安装与 assets：https://metadrive-simulator.readthedocs.io/en/latest/install.html
- Gymnasium 风格 reset/step：https://metadrive-simulator.readthedocs.io/en/latest/get_start.html
- 动作映射：https://metadrive-simulator.readthedocs.io/en/latest/action.html
- progressbar 2.5 发布文件：https://pypi.org/project/progressbar/
- panda3d-gltf 0.15.0（0.4.3源码要求 <1）：https://pypi.org/project/panda3d-gltf/0.15.0/

没有对正式算法效果新增结论；v2.1 的紧急制动回归与高碰撞率仍未解决。
