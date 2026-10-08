# RL AutoDrive Agent

**自动驾驶强化学习仿真与研究项目｜使用指南**

本项目提供轻量级 **native CPU 仿真**和可选的真实 **MetaDrive 0.4.3** 后端，支持 PPO、SAC、MAPPO、多种观测方式（state / semantic BEV / RGB + motion）、闭环性能评估、路线适配器以及安全监督相关研究。

> **当前版本：v4.2.0-dev（研究开发版）。** 这是源码仓库，不包含预训练 `.pt` 权重、MetaDrive assets、离线 wheelhouse 或大型实验输出。建议第一次使用时先完成第 1～4 步，无需安装 MetaDrive。
>
> **使用边界：仅限仿真研究，禁止用于实车控制。** 模拟环境中的成功率、低碰撞率不等于真实道路安全认证。

## 阅读导航

| 你想完成的任务 | 推荐章节 | 需要什么 |
| --- | --- | --- |
| 第一次安装并确认程序可运行 | 1～2 | Python 3.10+、pip |
| 从零训练并评估一个模型 | 3～4 | native 后端、CPU |
| 在真实 MetaDrive 中训练、评估 | 5 | MetaDrive 0.4.3 运行库和 assets |
| 复现 v3.2/v4.2 研究代码 | 6 | 原始模型权重与正式实验依赖 |
| 排查安装和运行问题 | 7 | 报错信息与运行报告 |

所有命令均在**项目根目录 `autodrive/`**运行。示例中的 `python` 指当前激活虚拟环境的解释器。

## 1. 获取项目并创建环境

准备 Python **3.10 或更高版本**（推荐 Python 3.11）与 Git。

```bash
git clone https://github.com/clonetimer/autodrive.git
cd autodrive
python -m venv .venv
```

**Windows PowerShell：**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS：**

```bash
source .venv/bin/activate
```

在 Windows 中，如果 PowerShell 不允许执行激活脚本，可跳过激活，并在后续命令中将 `python` 替换为 `.\.venv\Scripts\python.exe`，无需修改系统执行策略。

确认环境：

```bash
python --version
python -m pip --version
```

## 2. 安装基础依赖

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

基础环境包含 PyTorch、NumPy、PyYAML、Pillow 等，支持运行 CPU native 后端；**不需要 GPU、MetaDrive 或 Waymo 数据**。

可检查依赖一致性：

```bash
python -m pip check
```

具体依赖约束见 [`pyproject.toml`](pyproject.toml)；[`requirements-tested.txt`](requirements-tested.txt) 是历史测试环境的版本记录，并非所有平台通用的锁文件。

## 3. 在 native 后端训练第一个 PPO 模型

### 3.1 直接启动训练

```bash
python scripts/train.py --config configs/ppo_state.yaml --steps 4096 --output experiments/native_ppo_quickstart
```

参数说明：

| 参数 | 作用 |
| --- | --- |
| `--config configs/ppo_state.yaml` | 读取目前实际存在的单车 state PPO/native 配置 |
| `--steps 4096` | 将训练步数覆盖为 4096，适合首次链路验证 |
| `--output ...` | 训练结果与权重的输出目录 |

正常完成后，应生成：

```text
experiments/
  native_ppo_quickstart/
    model.pt             # 训练生成的模型权重
    ...                  # 训练配置、日志与结果记录
```

**4096 步只是用于确认真实训练、优化与保存链路可执行，不代表策略已经收敛。** 正式实验应增加训练预算、记录随机种子，并使用独立输出目录避免覆盖。

### 3.2 修改配置

主要配置文件在 `configs/` 下。可以复制 `configs/ppo_state.yaml` 并调整算法超参数、环境参数、随机种子等，再将新文件传入 `--config`。

可使用的其他算法入口及配置还包括 `configs/sac.yaml`、`configs/mappo.yaml`；不同算法/输入形式对环境和学习智能体数量存在约束，请先检查对应 YAML 与 [`docs/API.md`](docs/API.md)。

## 4. 评估模型并生成 GIF 演示

### 4.1 闭环测试

使用**刚刚训练生成的模型**，在测试种子上执行 10 回合评估：

```bash
python scripts/evaluate.py --model experiments/native_ppo_quickstart/model.pt --episodes 10 --split test --output reports/native_ppo_quickstart --trajectories
```

评估结果保存在 `reports/native_ppo_quickstart/`。重点检查成功率、碰撞、出界、路线完成度等指标；不能仅凭模型文件存在或训练正常退出断言驾驶策略有效。

### 4.2 导出轨迹 GIF

```bash
python scripts/demo.py --model experiments/native_ppo_quickstart/model.pt --seed 2000 --output reports/native_ppo_quickstart.gif
```

预期输出：

- `reports/native_ppo_quickstart.gif`：固定随机种子的实际仿真轨迹演示；
- `reports/native_ppo_quickstart.json`：同次演示的可核查轨迹记录。

### 4.3 运行基础测试

```bash
python -m pytest -q -m "not metadrive"
```

上述测试排除需要真实 MetaDrive 依赖/assets 的测试。请注意：**本 README 中的命令已与代码接口及配置文件进行静态核对，不代表在你的具体机器上全部运行通过。**

## 5. 使用真实 MetaDrive 0.4.3（可选）

native 是轻量运动学仿真，**不能代替真实 MetaDrive 运行**。MetaDrive 链路需要另行安装对应版本的模拟器、依赖及 assets；建议为此使用单独的 Python 虚拟环境。

### 5.1 安装 MetaDrive 相关依赖

在可以联网且系统依赖兼容的环境中，参考：

```bash
python -m pip install -r requirements-metadrive.txt
python -m metadrive.pull_asset
```

如依赖或 assets 无法自动获取，请参照：

- [`docs/METADRIVE_OFFLINE_v2.2.0.md`](docs/METADRIVE_OFFLINE_v2.2.0.md)：源码、assets 和 wheelhouse 的离线部署步骤；
- [`docs/METADRIVE_REAL_v2.3.0.md`](docs/METADRIVE_REAL_v2.3.0.md)：真实 MetaDrive CPU 训练与运行记录。

**仅安装 Python 包或成功 import 不等于仿真可用。**

### 5.2 运行真实仿真自检

```bash
python scripts/doctor.py --smoke --output reports/metadrive_doctor.json
```

`--smoke` 会尝试真实 MetaDrive reset/step（最多执行 40 步）。若失败，应检查诊断 JSON 中的缺失依赖、版本与资源问题；代码不会静默切换成 native 伪装成功。

### 5.3 训练与评估 MetaDrive PPO

确认上一步通过后，使用仓库中**实际存在**的 `configs/metadrive.yaml`：

```bash
python scripts/train.py --config configs/metadrive.yaml --steps 2048 --output experiments/metadrive_ppo_quickstart
python scripts/evaluate.py --model experiments/metadrive_ppo_quickstart/model.pt --episodes 5 --split test --output reports/metadrive_ppo_quickstart
```

这是一组小规模运行示例；真实驾驶性能需要独立的长训练和严格评估。

### 5.4 长训练：分段执行（Linux / WSL2）

```bash
python scripts/train_metadrive_segmented.py \
  --config configs/metadrive.yaml \
  --output experiments/metadrive_ppo_segmented \
  --total-steps 32768 \
  --segment-steps 4096 \
  --timeout 120
```

分段脚本通过独立子进程避免长时间运行时的部分模拟器生命周期问题，恢复**模型、优化器和训练计数器**，但**不会恢复进程间的物理世界状态**。该脚本使用 POSIX 进程组信号，不可假定其在原生 Windows 上直接适用。

对于 Linux 无头 RGB 相机实验，可能还需要配置 `DISPLAY` 或 `xvfb-run`。相关配置与限制参见 [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md)。

## 6. 复现已有模型与 v4.2 开发实验（可选）

当前 GitHub 是 **source-only** 仓库。`models/` 包含模型注册与说明文件，但**没有相应 `.pt` 权重**。

如需复现 v3.2 及 v4.2 路线，请先从之前的完整模型交付包取得以下两个文件并放入 `models/`：

```text
models/
  metadrive_ppo_state_capability_distilled_v28.pt
  metadrive_curve_route_adapter_v32.pt
```

这两个文件分别是冻结的 v2.8 base actor 和与其匹配的 v3.2 learned lateral adapter。务必核对 [`SOURCE_PACKAGE_CONTENTS.md`](SOURCE_PACKAGE_CONTENTS.md) 中的 SHA-256，不能随意替换不匹配的模型。

### 6.1 预览 v4.2 正式训练计划

确保已有完整模型和经自检的 MetaDrive 环境后，先执行**仅生成计划**的命令：

```bash
python scripts/run_v42_formal_authority.py \
  --base-model models/metadrive_ppo_state_capability_distilled_v28.pt \
  --lateral-adapter models/metadrive_curve_route_adapter_v32.pt \
  --output experiments/v4_2/formal \
  --dry-run
```

此命令不会启动训练；移除 `--dry-run` 才会开始 `G0` / `GPL` × 3 个独立种子的正式实验。

**实验数据隔离要求：** v4.2 正式训练仅使用 `authority_train`；`validation16` 与 `release19` 在正式冻结之前不可提前打开；已消费的历史验证/发布场景不得重新用于模型选择或调参。参见：

- [`docs/V42_VALIDATION16_PROTOCOL.md`](docs/V42_VALIDATION16_PROTOCOL.md)：预注册验证与晋升门禁；
- [`docs/V42_AUTHORITY_GATING_DEV_SELECTION.md`](docs/V42_AUTHORITY_GATING_DEV_SELECTION.md)：开发候选选择结果。

v4.2 仍为开发阶段，不得将 v3.2 的历史 release 数据当作 v4.2 的正式性能结论。

## 7. 常见问题

| 现象 | 原因与处理方式 |
| --- | --- |
| `model.pt` 找不到 | 本仓库没有预训练权重；先完成第 3 步生成自己的模型。 |
| `No module named ...` | 确认激活了正确的虚拟环境；重装对应 requirements，并执行 `python -m pip check`。 |
| `configs/ppo.yaml` 找不到 | 这是旧 README 的过期文件名；native PPO 请使用 `configs/ppo_state.yaml`。 |
| `configs/metadrive_ppo.yaml` 找不到 | 旧 README 的过期文件名；基本 MetaDrive PPO 请使用 `configs/metadrive.yaml`。 |
| MetaDrive 不能 reset/step | 运行 `python scripts/doctor.py --smoke`，核查模拟器版本、assets 与 Panda3D 运行库。 |
| 服务器 RGB 渲染失败 | Linux 无头运行可能需要 `xvfb-run`；首先用 state 模式验证非相机链路。 |
| 训练分段脚本在 Windows 出错 | 该脚本使用 POSIX 进程组控制；建议在 Linux/WSL2 中执行。 |
| 修改 README 后完整性校验失败 | 原 `MANIFEST.sha256` 是旧版发行内容快照；需要同步更新 `README.md` 的 SHA-256 记录。 |

## 8. 目录结构与更多资料

```text
autodrive/
├── configs/               # 算法、环境、场景和奖励配置
├── src/                   # 强化学习、仿真、导航、安全与评估源码
├── scripts/               # 训练、评估、可视化和实验工具
├── tests/                 # 单元测试、集成测试与实验协议测试
├── docs/                  # 架构、研究记录、验收与已知限制
├── models/                # 注册信息；源码发行不包含 .pt 权重
├── experiments/           # 训练实验输出位置
├── reports/               # 评估报告和图像输出位置
├── requirements*.txt      # 依赖文件
└── pyproject.toml         # Python 项目元数据
```

详细资料：

- [系统架构](docs/architecture.md)
- [API 使用说明](docs/API.md)
- [已知限制与边界](docs/KNOWN_LIMITATIONS.md)
- [研究版本更新记录](docs/CHANGELOG.md)
- [v3.2 曲线跟踪研究](docs/CURVE_ROUTE_ADAPTER_v3.2.0.md)
- [v4.2 正式验证协议](docs/V42_VALIDATION16_PROTOCOL.md)

旧 README 的长篇实验历史可从 Git 历史版本查阅；本 README 以**用户实际操作顺序**组织。文档中的命令不保证模型已经收敛、结果可直接复现或能够用于实车。

---

**License：MIT**（以仓库根目录 [`LICENSE`](LICENSE) 为准）。
