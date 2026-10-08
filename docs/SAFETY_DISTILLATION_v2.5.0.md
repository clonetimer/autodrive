# v2.5.0 Safety Policy Distillation — 实验与验收记录

## 目标

v2.4.0 已证明独立 `SafetySupervisor` 能在有限 MetaDrive holdout 上把组合系统碰撞降到 0，但介入时间接近一半。v2.5.0 不再把“安全层能兜底”作为终点，而是尝试把安全层的纵向修正蒸馏回神经策略，目标是：

1. raw learned policy 自身降低碰撞；
2. 在仍保留 supervisor 的组合系统里，减少规则层介入强度和介入时间；
3. raw policy 与 supervised system 继续分开评估，不把系统级结果包装成策略安全能力。

这仍然是研究实验，不是 CBF/MPC 形式化安全证明，也不用于实车。

## 方法

源模型固定为 `models/metadrive_ppo_state_32768.pt`，其原始 RL 训练步数为 32,768 world steps。蒸馏阶段不更新 critic 和 exploration `log_std`，只更新 state encoder + actor。

### 真实 MetaDrive 采集

使用困难一些的 MetaDrive 分布：

- `traffic_density=0.18`
- `max_steps=300`
- reward v4 / safe profile（仅用于环境记录，蒸馏目标直接来自动作）
- supervisor: `ttc_threshold=1.8s`, `risk_threshold=0.68`
- collection seeds: **8000–8011**，与 release4 完全互斥

每一步由当前冻结 actor 提议动作，`SafetySupervisor` 在同一个真实物理状态上决定实际执行动作。训练 target 为 `applied_action`。supervisor 从不修改 steering，因此 steering target 与原 actor 相同；主要学习信号来自 longitudinal correction。

实际采集：

- 1,419 transitions
- 733 intervention transitions
- intervention fraction **51.66%**
- mean longitudinal correction **0.4127**
- p95 correction **0.9229**
- supervisor 下采集过程 **0 collision**
- 12/12 collection seeds 到达终点

数据：`experiments/v2_5/safety_distill_round1/safety_distillation_dataset.npz`。

### Intervention-aware weighted distillation

loss 对所有透明样本保留 unit-weight 约束，以避免无风险区域策略无缘无故漂移；对 supervisor 真正介入的样本按 severity 增权。由于 supervisor 不改 steering，steering component 权重较低但不为 0，用作行为保持正则。

候选超参数只在 `validation=1000–1009` 10 个开发场景上选择，**release4 在候选冻结前没有被评估**。四个候选都在这 10 个场景达到 100% success / 0 collision；最终选择 `intervention_weight=0.75, epochs=8`，因为它在满足安全开发门槛后平均速度最高（约 8.91 m/s）。

最终模型：`models/metadrive_ppo_state_safety_distilled_v25.pt`。

## release4 冻结测试

release4 = **7000–7039**，共 40 个与训练/开发划分互斥的 MetaDrive 场景。候选选定后只做冻结评估，不再据此回选权重。

| System | Success | Collision | Out-of-road | Avg speed | Supervisor intervention |
|---|---:|---:|---:|---:|---:|
| Base raw PPO | 42.5% | 57.5% | 0% | 10.58 m/s | 0% |
| **Distilled raw PPO** | **60.0%** | **40.0%** | 0% | 8.81 m/s | 0% |
| Base PPO + supervisor | 100% | 0% | 0% | 8.95 m/s | 47.22% |
| **Distilled PPO + supervisor** | **100%** | **0%** | 0% | 8.17 m/s | **27.11%** |

主要变化：

- raw success：+17.5 percentage points；
- raw collision：−17.5 percentage points；
- supervised intervention fraction：47.22% → 27.11%，相对下降约 **42.6%**；
- mean longitudinal correction：0.3825 → 0.1435，相对下降约 **62.5%**；
- 40 个场景中有 **38 个** supervisor 介入比例下降、2 个上升；
- raw longitudinal jerk mean abs：4.35 → 1.35 m/s³，但平均速度也下降约 1.77 m/s。

配对 raw outcome：10 个场景由失败变成功，3 个由成功变失败，14 个都成功，13 个都失败。双侧 exact McNemar 描述性 p≈0.0923；样本量太小，不能据此宣称统计显著的安全改善。

机器可读结果：`reports/v2_5/comparison.json`；逐回合原始报告位于 `reports/v2_5/release4/`。

## Waymo/ScenarioEnv 小样本回归

最终 curated distilled raw policy 在 assets 自带的 3 个 Waymo 转换场景上重新做真实 `ScenarioEnv` 闭环 ego 控制：3/3 到达、0 collision，平均速度约 16.58 m/s；证据位于 `reports/v2_5/waymo_distilled_curated/`。样本只有 3 个，而且背景交通不是完整反事实响应，因此只能证明接口回归，没有资格称为真实数据泛化。

## 为什么仍然不能说“策略已经安全”

1. raw distilled policy 在 release4 仍有 **40% collision rate**；
2. supervised system 的 0 collision 依赖启发式规则层，且仍有 27.11% 时间介入；
3. source PPO 主要来自单训练 seed，缺少训练种子不确定性；
4. supervisor 使用仿真真值风险信息生成标签，蒸馏属于 privileged safety teacher imitation；
5. 40 个 procedural scenarios 远小于真实安全验证需要的场景覆盖；
6. 没有传感器失效、现实域差异、HIL/SIL法规流程或形式化安全证明。

## 可复现命令

```bash
python scripts/distill_metadrive_safety.py \
  --source models/metadrive_ppo_state_32768.pt \
  --collection-config configs/metadrive_supervised_eval.yaml \
  --output experiments/my_safety_distill \
  --iterations 1 --seeds-per-iteration 12 --max-steps-per-seed 300 \
  --epochs-per-iteration 8 --batch-size 256 \
  --learning-rate 2e-4 --intervention-weight 0.75 --steering-weight 0.35
```

冻结 raw / supervised 对照：

```bash
python scripts/evaluate.py --model models/metadrive_ppo_state_safety_distilled_v25.pt \
  --episodes 40 --split release4 --output reports/distilled_raw_release4

python scripts/evaluate.py --model models/metadrive_ppo_state_safety_distilled_v25.pt \
  --config configs/metadrive_supervised_eval.yaml \
  --episodes 40 --split release4 --output reports/distilled_supervised_release4
```

### 演示

- `videos/v2_5/distilled_raw_success_seed7000.gif`：raw success；
- `videos/v2_5/distilled_raw_collision_seed7002.gif`：raw collision，失败不隐藏；
- `videos/v2_5/distilled_supervised_seed7002.gif`：同 seed 开启 supervisor 后的组合系统。

## 下一步

release4 已经被查看，因此之后若继续调参必须启用新的冻结 split，不能把 release4 重复当成最终测试集。下一阶段优先：多训练 seed 的 source PPO、第二轮 DAgger-style 聚合、显式风险特征/时序状态、emergency/follow/merge/yield 课程，以及 supervisor intervention penalty 的 shielded RL 对照。
