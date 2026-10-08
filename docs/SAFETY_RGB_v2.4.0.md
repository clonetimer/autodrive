# v2.4.0 — Safety supervision and real RGB learning path

## 1. 目标

本版本针对 v2.3 的两个主要问题做实质性修改：

1. state PPO 在更难随机种子上的碰撞率高，需把**策略本身**与**系统级安全补偿**分开验证；
2. v2.3 只验证了 RGBCamera sensor，没有可训练 raw-RGB policy。

## 2. Reward v4

`src/envs/reward.py` 保留 v2/v3，用 v4 新增有限时域安全项：

- `ttc`：4秒内 TTC 连续惩罚；
- `closing`：按 front risk × closing speed 约束持续逼近；
- `unsafe_accel`：危险状态下正加速度惩罚；
- `emergency_brake`：风险状态下合理制动的小额 credit，避免 comfort 项反向压制紧急制动。

所有稠密项按真实 `dt` 缩放；terminal collision/offroad/success 不按 `dt` 缩放。固定单步反例测试确认 v3/v4 不再让 2×目标速度仅靠 progress 获得更高净收益。

短期 transfer/KL 尝试没有得到可重复的 learned-safety 改善，因此**不进入精选模型和发行结论**。v4 当前是已实现、已测试的目标函数，不是“已证明更安全的策略”。

## 3. 独立 SafetySupervisor

`src/safety/supervisor.py` 是一个有状态纵向动作过滤器：

- 输入 actor 提出的 `[steer, signed_pedal]` 与真实仿真 snapshot；
- 只修改第二维 longitudinal command，从不改 steering；
- 基于投影到道路切向的 gap、closing speed、TTC、future lateral overlap 和 safe target speed；
- 采用 warning/critical 分级、hysteresis、release steps 与 throttle recovery slew；
- 每步公开 `proposed/applied/intervened/reason/severity/risk`，评估输出介入次数和时间占比。

它不是 neural policy，也不是隐藏的后处理。任何 supervised 结果必须单独标注。

### 冻结 release3 结果

同一个 32k state PPO 权重，不重新训练或挑 checkpoint；40 个 `release3` 场景（seed 6000–6039）：

| 指标 | raw PPO | PPO + supervisor |
|---|---:|---:|
| success | 42.5% | **100%** |
| collision | 57.5% | **0%** |
| off-road | 0% | 0% |
| mean route completion | 79.95% | 95.93% |
| avg speed | 10.28 m/s | 8.86 m/s |
| intervention time fraction | 0% | **45.82%** |
| interventions | 0 | 2,084 |

因此结论是：**组合系统在这 40 个场景没有发生碰撞，但近半时间发生纵向介入；原始策略本身明显不安全。**

release2 的独立 40 场景得到同向结果：raw 42.5% success / 57.5% collision；supervised 100% / 0%，介入约 46.67%。

## 4. Raw RGB + ego/navigation motion

### 输入

`rgb_ego` observation：

- `rgb`: uint8 `[3,H,W]`，来自真实 MetaDrive `RGBCamera`；
- `motion`: float32 `[20]`，包含 ego 速度/加速度/yaw rate/steering、目标速度、车道/航向误差和局部 route point。

`RGBEncoder` 使用小型 CPU CNN，之后与 motion MLP 特征融合，再进入 PPO actor/critic。没有 semantic BEV 或邻车真值特征注入 RGB branch，但 motion/navigation 是显式辅助输入，所以不是 camera-only。

### 真实 PPO smoke

实际训练 384 world steps：

- 模型 SHA-256: `2558b2068fa4931b5c751f54d1fd4867e5d0846838332a9785d22f98759c04dd`；
- RGB encoder 参数变化 L1 ≈ 90.82；
- 40-step 新 seed smoke：route completion ≈ 4.72%，avg speed ≈ 0.05 m/s，timeout；
- 结论：**真实像素 PPO 反传链路成立，但策略几乎停驶，不具备驾驶能力。**

## 5. Privileged teacher → RGB actor distillation

为了不把大量 CPU 时间浪费在像素随机探索上，新增 `src.agents.distillation` 和 `scripts/distill_metadrive_rgb.py`：

1. frozen state PPO teacher 读取同一物理快照的 state；
2. teacher action 实际控制 MetaDrive；
3. student 只记录真实 RGB + motion；
4. actor/encoder 用 deterministic action MSE 训练；critic 和 exploration `log_std` 不训练；
5. 训练证据保存 teacher hash、样本 seed、MSE 和 student hash。

当前 curated experiment 使用 6 个 train seed、48 个真实像素样本：

- initial MSE = 0.371472；
- final MSE = 0.00015036；
- actor+encoder 参数变化 L1 ≈ 376.03；
- 40-step 新 validation seed smoke：route completion ≈ 20.83%，avg speed ≈ 4.93 m/s，仍 timeout。

因此 BC 改善了短程运动，但样本太少，**不能称为会驾驶的 RGB agent**。一次尝试扩大 RGB 数据收集出现运行时超时，已移动到 `dev_smoke`，不进入发行统计。

## 6. 下一验收门槛

下一轮不以“版本号”作为目标，而采用能力门槛：

1. **Raw state policy**：至少 3 training seeds；固定 unseen split 上降低碰撞；
2. **Supervisor**：在新的 untouched split 保持低碰撞，同时将 intervention fraction 明显低于当前约 46%；
3. **RGB BC**：扩展到至少数百个多场景样本，独立验证 seed 的 action error 与闭环进度同时改善；
4. **RGB PPO fine-tune**：从 BC student 开始做真实 MetaDrive PPO，评估 success/collision/offroad，而不是只看 loss；
5. **复杂场景**：emergency/follow/merge/yield 分开统计，不允许简单 cruise 掩盖安全失败。
