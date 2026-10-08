# RL Autonomous Driving Agent · v2.0.0

**CPU 可运行研究交付版，不是“全部原始目标验收通过”的生产发布。**

本仓库已经包含实际更新参数的 PPO、SAC、参数共享 IPPO / MAPPO，车辆共享同一物理世界的多智能体仿真，BEV CNN，以及模型训练、恢复、闭环评估、演示、日志和可复现测试。所有训练权重与指标均来自本次沙盒实际运行；并非随机初始化权重或手工填写的示例结果。

**必须区分两个后端：** `native` 是本项目实现的 CPU 运动学交通模拟器；`metadrive` 是调用真正 MetaDrive/Panda3D 的适配器。沙盒缺少 Gymnasium、Panda3D、pygame、MetaDrive 运行依赖与 assets，且软件包下载失败。因此 MetaDrive 真正训练目标仍为阻塞，相关测试明确跳过，绝不把 native 的训练写成 MetaDrive 实验。证据见 `reports/metadrive_doctor.json`、`experiments/metadrive_attempt/failure.json`、`logs/dependency_install_failed.log`。

## 1. 已交付范围

| 目标 | 代码与运行结果 | 不能据此声称 |
|---|---|---|
| v1.1 MetaDrive 实验 | 真实环境构造、动作/观测适配、单/多智能体、终态捕获、安装与诊断脚本；实际尝试启动后记录依赖阻塞 | 已跑过 MetaDrive、已通过真实仿真训练验收 |
| v1.2 MARL | 同一道路世界多个可学习车辆；CTDE MAPPO 和参数共享 IPPO；实际 CPU 更新与保存权重 | 已做全量 MAPPO 原论文复现、已实现 V2X 协议 |
| v1.3 BEV | 五通道语义 BEV，CNN 学习，state/bev/hybrid 模式，真实梯度与闭环结果 | 原始相机/激光雷达感知、多传感器融合标定 |
| v2.0 集成 | 语义输入 → 可学习编码器 → 策略 → 连续控制；训练/评估/回放/归档 | 原始 RGB 到驾驶控制的量产级端到端系统、真实道路安全性 |

结果以 `reports/RELEASE_STATUS.json`、`docs/experiment_report.md` 与原始 CSV 为准。单元测试通过不代表驾驶成功率达标。

## 2. 安装

核心运行只需要 Python、NumPy、PyTorch、PyYAML、Pillow、OpenCV，不强制依赖 GPU、Gymnasium、SB3 或 TensorBoard。沙盒实测 Python 3.13.5 + PyTorch 2.10.0 CPU；完整版本见 `reports/runtime_manifest.json`。

```bash
python -m venv .venv
# Linux/macOS；Windows 使用 .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

离线沙盒已经具备核心依赖，可直接在仓库根目录运行下列脚本，无需重新下载。可选依赖的真实集成测试在依赖不齐全时显示 `SKIPPED`，不是 `PASSED`。

## 3. 训练、恢复、评估

```bash
# 单智能体 PPO；输出目录必须是新的目录，防止覆盖实验。
python scripts/train.py --config configs/ppo_state.yaml --output experiments/my_ppo

# 两个车辆共享世界，中心化 critic、去中心化 actor。
python scripts/train.py --config configs/mappo.yaml --output experiments/my_mappo

# 纯语义 BEV 输入，CNN 与 PPO 联合更新。
python scripts/train.py --config configs/ppo_bev.yaml --output experiments/my_bev

# 多智能体 + BEV/状态融合 + 随机道路/观测噪声。
python scripts/train.py --config configs/end_to_end.yaml --output experiments/my_system

# PPO/MAPPO 恢复：--steps 指训练计数器的最终目标，而非额外增加步数。
python scripts/train.py --config configs/ppo_state.yaml \
  --resume experiments/my_ppo/model.pt --steps 65536 --output experiments/my_ppo_resumed

# 测试集固定种子从 2000 开始；不拿 test 结果选超参数。
python scripts/evaluate.py --model experiments/release/ppo_state_seed7/model.pt \
  --episodes 20 --output reports/my_eval --trajectories

python scripts/benchmark.py --model experiments/release/ppo_state_seed7/model.pt \
  --episodes 20 --output reports/my_generalization

python scripts/demo.py --model experiments/release/mappo_hybrid/model.pt \
  --seed 2000 --output videos/my_system.gif
```

`total_steps` 是 **joint world steps**。多智能体的有效样本数另外记录在 `active_agent_steps_this_run`，不会把两者混为一谈。成功/碰撞/出界导致 `terminated`；外部时间上限导致 `truncated`。GAE 在截断时 bootstrap，但不会跨 reset 泄漏优势值。

恢复会恢复模型、优化器、计数器与 Torch 随机状态，但会从新的 episode 边界开始，不承诺完整世界快照级逐帧续跑。SAC 模型用于推理；本版本**明确拒绝**伪装成完整 SAC 断点恢复的 actor-only 恢复。

## 4. 查看产物

PPO/IPPO/MAPPO 的每个 `experiments/release/<run>/` 包含 `config.yaml`、`runtime.json`、`model.pt`、`result.json`、`training.csv`、`episodes.csv`、`tensorboard/` 及阶段检查点；SAC 产物保留最终 actor、训练日志与结果，不承诺完整SAC恢复。`result.json` 包含权重 SHA-256、参数变化量、真实步数与用时。

```bash
# TensorBoard UI 是可选安装；事件文件由公共 protobuf/TFRecord 格式写入。
python -m pip install tensorboard
tensorboard --logdir experiments/release

# 完整复现实验矩阵；不会默默复用旧结果。
python scripts/run_experiments.py --output experiments/reproduced

# 用复现的训练输出生成独立评估（验证集与测试集固定）
python scripts/evaluate_release.py --training experiments/reproduced --output reports/reproduced

# 可测量门槛驱动的课程；本次真实运行没有达到升级门槛
python scripts/train.py --config configs/curriculum.yaml --output experiments/my_curriculum

# 测试与覆盖率
python -m pytest -q --cov=src --cov-report=term-missing --cov-report=html:reports/coverage_html
```

`reports/release_evaluation/` 存放所有逐回合评估，不删碰撞样本、不把视频挑选结果当评估集。GIF 旁边有对应 JSON 轨迹。`docs/experiment_report.md` 解释样本量、短训练预算、失败案例与算法比较的限制。

## 5. 真正的 MetaDrive

上传的 `metadrive.zip` 是 0.4.3 源码，不包含运行所需的匹配 assets，也不等于已经安装 Panda3D。使用有网络的 CPU 机器或离线 wheelhouse：

```bash
# 网络安装；pip 失败会非零退出，而不是切换为 native。
python scripts/bootstrap_metadrive.py --source-zip /path/to/metadrive.zip --workspace external

# 离线安装：wheelhouse 必须包含当前 Python/系统匹配的全部依赖 wheel。
python scripts/bootstrap_metadrive.py --source-zip /path/to/metadrive.zip \
  --workspace external --wheelhouse /path/to/wheels --assets-zip /path/to/assets.zip

python scripts/doctor.py --smoke --output reports/my_metadrive_doctor.json
python -m pytest -q -m metadrive
python scripts/train.py --config configs/metadrive.yaml --output experiments/real_metadrive
python scripts/train.py --config configs/metadrive_mappo_bev.yaml --output experiments/real_metadrive_mappo
```

MetaDrive 同进程引擎生命周期必须显式 `close()`。多智能体适配在车辆移除前捕获真实终态；使用 simulator 原始奖励，报告里不将其与 native reward 数值直接比较。当前该实现做过源码/API 对照，尚未在本沙盒运行通过；首次联网执行仍可能暴露版本适配问题。

## 6. 数据与传感器边界

`data/synthetic_replay.json` 明确是合成测试轨迹，**不是 nuScenes 或 Waymo**。下载器要求 HTTPS、许可确认、预期 SHA-256；不会绕过需要注册的数据集权限。CSV 规范为 `time_s,track_id,x_m,y_m,yaw_rad,speed_mps`。

```bash
python scripts/download_dataset.py --url "https://your-authorized-source/data.zip" \
  --sha256 <64-character-expected-digest> --accept-license --output data/input.zip
python scripts/prepare_dataset.py --csv data/tracks.csv \
  --provenance "dataset name, license and source" --output data/scenario.json
```

本版本提供可执行的通用下载/校验、CSV 转换和非响应式轨迹回放，不提供完整的原始 nuScenes/Waymo 格式转换。真实 ScenarioNet 数据可通过 MetaDrive `ScenarioEnv` 适配路径接入，但尚未验证。**ScenarioNet pickle 只能来自可信来源，加载 pickle 存在任意代码执行风险。**

BEV 来源于模拟器真值几何，通道是道路、车道边界、参考路线、其他车辆、自车。没有虚构 Camera/LiDAR 感知网络或 learned world model。推理没有隐藏规则控制器代替神经策略纠错。

## 7. 文档与验收

架构和接口：`docs/architecture.md`、`docs/API.md`。算法与边界：`docs/algorithm.md`、`docs/KNOWN_LIMITATIONS.md`。开发台账：`docs/development_log.md`。需求对应：`docs/acceptance_matrix.md`。完整文件目录：`docs/TREE.txt`。完整性校验：`MANIFEST.sha256` 与 `scripts/verify_package.py`。

仓库是研究与校招项目基础，不应直接用于实车控制。简历应准确写为“实现并在自研 CPU 仿真中训练验证”，尚不能写“完成 MetaDrive 真实训练实验”或引用不存在的实车安全结果。
