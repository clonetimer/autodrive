# RL Autonomous Driving Agent · v2.1.0

**CPU 仿真研究交付版：实际修改、训练、评估和归档；不是安全驾驶或全部原始目标验收通过的生产版本。**

v2.1.0 在原 v2.0.0 上修复奖励方向、补充 BEV 运动信息与历史，加入完整 SAC 学习状态恢复和 MetaDrive 安装前检查。新版本实际执行 241,664 个联合世界步，完成 10 次新训练和 1 段续训；不是只有配置或随机权重。工程测试与驾驶能力分别验收：巡航、密集交通改善，但紧急制动退步，不能声称整体更安全。

当前有两个明确区分的后端：`native` 为本项目 CPU 自行车动力学交通模拟器，`metadrive` 为真正调用 MetaDrive 的适配器。所有新增训练均来自 native。MetaDrive 本轮再次检查发现 DNS 失败、缺少运行依赖与 assets，真实执行步数仍为 0，没有伪装或自动降级。

## 1. 新版本结果

三种训练种子 7/8/9，每种奖励每种子 32,768 步；每场景类型使用相同 40 个开发验证种子 3000–3039，均使用最后检查点。百分比为三训练种子的均值。

| 场景 | 旧奖励成功率 | 新奖励成功率 | 旧奖励碰撞率 | 新奖励碰撞率 |
|---|---:|---:|---:|---:|
| 巡航 | 66.67% | 85.83% | 32.50% | 10.00% |
| 密集交通 | 14.17% | 56.67% | 85.83% | 43.33% |
| 紧急制动 | 25.00% | **14.17%** | 73.33% | **85.83%** |

IDM 在这三个有限场景集合中成功率均为 100%，仍优于 RL。单帧 BEV+motion PPO 成功率 92.5%，四帧版本 77.5%，不能宣称帧越多效果越好。两车 MAPPO 四帧工程验证每车成功率 92.5%，**全车共同成功率 85%**（20 场景），不是与历史 MAPPO 不同场景结果的公平排名。

完整证据与限制见 `docs/IMPROVEMENT_REPORT_v2.1.0.md`、`reports/v2_1/evaluations/`、`reports/RELEASE_STATUS.json`。既有 2000 系列场景已用于开发，不再称为独立最终测试集；预留 7000–7199 本轮未使用。

## 2. CPU 安装与测试

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python scripts/verify_package.py
python -m pytest -q
```

核心无需 CUDA、Gymnasium、SB3 或 TensorBoard；依赖见 `requirements.txt`。沙盒实际版本见 `requirements-tested.txt`、`reports/v2_1/runtime_validation.json`。具备依赖的离线环境可直接运行仓库脚本。真实 Gymnasium/MetaDrive 检查有 5 项跳过，**不是通过**。全局预装 moviepy 与 Pillow 存在无关依赖冲突，本项目不使用 moviepy；不声称全局环境完全兼容。

## 3. 已训练模型与新训练

```bash
# 控制性复核：旧超速缺陷、新奖励方向、BEV 速度区分
python scripts/audit_improvements.py --output reports/my_audit.json

# 已交付新奖励模型；验证种子只是开发集，不称为最终测试
python scripts/evaluate.py \
  --model experiments/v2_1/ppo_reward_v3_seed7/model.pt \
  --split validation --episodes 20 --output reports/my_evaluation

# 单帧语义 BEV + 20 维自车运动/导航；不含邻车真值速度
python scripts/train.py --config configs/improvement/bev_ego.yaml --output experiments/my_bev_ego

# 四帧 ego-motion 补偿历史，CNN + motion + 帧有效性/时间
python scripts/train.py --config configs/improvement/bev_temporal.yaml --output experiments/my_temporal

# 同一世界两辆学习车，集中 critic / 分散 actor
python scripts/train.py --config configs/improvement/mappo_temporal.yaml --output experiments/my_mappo

# 明确使用新奖励，或使用 ppo_reward_v2.yaml 重现旧奖励
python scripts/train.py --config configs/improvement/ppo_reward_v3.yaml --output experiments/my_ppo

# 固定种子实回放，GIF 旁保存同源 JSON 轨迹
python scripts/demo.py --model experiments/v2_1/ppo_bev_ego_frames4/model.pt \
  --seed 3000 --output videos/my_temporal.gif
```

输出目录必须为新的目录，避免覆盖原结果。`world_steps` 是联合物理世界步；多智能体的有效 agent 样本另计。`reward_version: 3` 为新配置默认值，旧模型缺失该字段时载入为 2，防止悄悄改变历史评估。旧 YAML 在新代码默认奖励已变更，重现旧训练应明确设置 `reward_version: 2`。

## 4. SAC 真实续训

```bash
python scripts/train.py --config configs/improvement/sac_resume.yaml \
  --steps 2048 --output experiments/my_sac
python scripts/train.py --config configs/improvement/sac_resume.yaml \
  --resume experiments/my_sac/training_state.pt \
  --steps 4096 --output experiments/my_sac_resumed
```

`--steps` 是恢复后的累计目标，第二条只再执行 2048 步。`model.pt` 仅用于推理；完整恢复必须使用 `training_state.pt` 或 `.resume.pt`，内含 actor、双 Q、target Q、温度、全部优化器、replay、随机状态和计数。单元测试验证恢复后的下一优化步逐张量完全一致；物理世界没有序列化，恢复会开始新回合，不是原驾驶轨迹逐帧续跑。

## 5. 重现实验与评估

```bash
python scripts/run_v21_experiments.py --output experiments/my_v21_matrix
python scripts/evaluate_v21.py --models experiments/my_v21_matrix \
  --section state --output reports/my_v21
python scripts/evaluate_v21.py --models experiments/my_v21_matrix \
  --section vision --output reports/my_v21
python scripts/performance.py --steps 500 --repeats 2 \
  --modes state bev_ego --bev-frames 4 --output reports/my_perf.json
python scripts/validate_v21.py --output reports/my_cli_check.json
python -m pytest -q --cov=src --cov-report=term-missing
```

计划先于执行写出，最后检查点固定，不拿结果回选模型。失败评估保存末 4 秒动作、奖励分项、风险和终态。纵/横向舒适性指标有单位、时间权重、RMS/p95/峰值；缺失数据为 null 而非伪造零。Replay 图像保持 uint8，默认 512 MiB 容量上限；多帧图像 SAC 必须合理降低 replay_size。

## 6. 解除真实 MetaDrive 安装阻塞

详见 `docs/METADRIVE_OFFLINE_v2.1.0.md`。上传源码为 0.4.3，不含 assets。先做不修改环境的检查：

```bash
python scripts/bootstrap_metadrive.py --source-zip /path/to/metadrive.zip \
  --preflight-only --report reports/my_preflight.json
```

联网机器准备匹配目标 Python/操作系统/架构的 wheelhouse，再提供匹配源码版本的 assets。脚本检查文件清单、哈希、wheel 标签、assets 版本；安装必须在隔离虚拟环境中，缺依赖或网络时非零退出，不切换 native。离线镜像实际构建、MetaDrive step、SB3 参考训练仍未通过本沙盒验证。

## 7. 项目边界与文档

BEV 是模拟器真值语义栅格，不是原始 RGB 重建；新运动信息与历史不能冒充真实传感器感知。MARL 为参数共享 IPPO/CTDE MAPPO，没有 PettingZoo 官方认证、通信学习、RNN 或世界模型。现有通用授权下载/校验、CSV 转换与合成轨迹回放未增加真实 nuScenes/Waymo 实验。

当前接口：`docs/API.md`；算法：`docs/algorithm.md`；开发台账：`docs/development_log.md`；需求验收：`docs/acceptance_matrix.md`；限制：`docs/KNOWN_LIMITATIONS.md`；完整路径：`docs/TREE.txt`。`docs/experiment_report.md` 和 `experiments/release/` 保留 v2.0 历史结果；当前结果在 `reports/v2_1/`。模型索引含历史及新增模型，`MANIFEST.sha256` 覆盖发布文件，不是数字签名。

工程包可复核并不代表策略安全。本项目不得直接用于实车控制；不能在简历中写成“已完成 MetaDrive 真实训练”或“已实现原始相机端到端自动驾驶”。
