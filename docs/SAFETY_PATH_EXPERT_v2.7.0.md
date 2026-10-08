# v2.7.0 — Privileged lateral expert, path diagnostics and release6 freeze

## 1. 研究问题

v2.6 在同一纵向 `SafetySupervisor` 下把 collision 大幅过滤后，`curve_dense` / `mixed_unseen` 仍主要以 off-road 失败，说明瓶颈已经包含横向/路径跟踪。v2.7 不继续把 longitudinal safety label 反复蒸馏，而是建立**训练期特权 lateral expert → neural policy** 的可审计实验链路。

数据域严格分离：

- `expert_train = 10000–10999`：仅用于特权 expert 数据采集；
- `validation2 = 11000–11199`：候选排序与预声明 promotion gate；
- `release6 = 12000–12199`：模型冻结后才打开；本版本实际使用 12000–12039。

release4/release5 均已在历史版本查看，不再用于本轮调参。

## 2. 实质性实现

### 2.1 MetaDrive IDM privileged steering teacher

`src/experts/metadrive_idm.py` 将 MetaDrive 内置 `IDMPolicy` 绑定到 ego 车辆，在真实 simulator state 上生成 expert action。teacher 可以读取 MetaDrive 内部导航/感知信息，因此明确属于 **privileged training teacher**。最终 raw student 部署只读取项目标准 state observation，teacher 不随模型发布执行。

正式两条路线：

| 路线 | expert_train seeds | transitions | 训练范围 | 结果 |
|---|---:|---:|---|---|
| `expert_steering` | 10000–10015 | 3,485 | encoder冻结，仅 steering actor row | validation2 未改善 |
| `expert_balanced` | 10032–10047 | 3,529 | encoder + actor，critic/log_std冻结 | validation2 排名第一并通过工程 promotion gate |

正式路线合计 7,014 条真实 MetaDrive expert transitions、32 个互斥 training seeds。

### 2.2 Path suite

新增 `src/evaluation/path_suite.py`，把 lateral competence 从高密度拥堵中单独剥离：

- `curve_sparse`: `SCS`, density 0.05;
- `curve_moderate`: `SCS`, density 0.15;
- `mixed_sparse`: `SCXTO`, density 0.05;
- `mixed_moderate`: `SCXTO`, density 0.15.

Path suite 可以与同一个 longitudinal supervisor 组合，使“刹车/跟车过滤”尽量不掩盖 steering/path failure。它仍不是独立形式化安全测试。

### 2.3 Anchored expert distillation 负实验

真实实验发现：扩大 encoder+actor imitation 虽然可继续降低 teacher MSE，但容易造成 source representation drift。新增 `src/experts/distillation.py`，同时加入 frozen source action anchor 与 feature anchor，并修复离线 distillation 每个 minibatch 重复重新 permutation、导致一个 epoch 可能重复/遗漏样本的问题。

两个 anchored 候选都降低 teacher MSE，并把 source-action MSE 控制在有限范围，但 validation2 path success 为 0%，没有超过 `expert_balanced`。因此 anchoring 作为**负/机制实验**保留，不回选为最终策略。

## 3. Validation2：选择与冻结

20 个固定 safety-suite 场景（11000–11019）：

| Candidate | Success | Collision | Off-road | Catastrophic | Route completion |
|---|---:|---:|---:|---:|---:|
| v2.5 base | 15% | 85% | 0% | 85% | 42.10% |
| steering-only expert | 10% | 85% | 5% | 90% | 41.55% |
| **balanced expert** | **20%** | **75%** | **5%** | **80%** | **46.43%** |
| anchored medium | 20% | 80% | 0% | 80% | 43.31% |

固定排序选择 `balanced`。相对 base：success +5pp、collision -10pp、off-road +5pp、catastrophic -5pp。它恰好满足预声明 gate 的 `success_gain >= 5pp` 与 off-road 上限，因此 **promotion=true**。

这个 gate 的“material gain”实际只对应 20 个 validation 场景里的**一个场景**，因此它只是工程晋升门槛，不是统计显著性。`reports/v2_7/FROZEN_MODEL.json` 在打开 release6 前固定模型 SHA 与 selection SHA。

## 4. Release6：冻结后一次性诊断

release6 safety suite 使用 12000–12039 共 40 场景。

| System | Success | Collision | Off-road | Timeout | Route completion | Supervisor intervention |
|---|---:|---:|---:|---:|---:|---:|
| v2.5 source raw | 15.0% | 77.5% | 7.5% | 0% | 42.41% | 0% |
| v2.7 expert-distilled raw | **17.5%** | 80.0% | **2.5%** | 0% | **44.97%** | 0% |
| v2.5 source + supervisor | 25.0% | 0% | 50.0% | 25.0% | 54.70% | 51.47% |
| v2.7 expert-distilled + supervisor | 22.5% | 7.5% | **32.5%** | 37.5% | 54.45% | **25.36%** |

Raw policy 的 paired success 为 2 改善 / 1 退步，exact McNemar p=1.0。collision-or-off-road 从 85.0% 降到 82.5%，仅 -2.5pp。**release6 没有确认 validation2 上的策略安全改进。** 更具体地说，selected raw 把一部分 off-road 转回 collision，同时成功率只增加 2.5pp。

同一 longitudinal supervisor 下，规则介入时间从 51.47% 降到 25.36%（相对约 -50.7%），但系统 success 25%→22.5%，且 selected system 出现 7.5% collision。因此不能把“规则依赖下降”解释为“系统安全提高”。

## 5. Secondary path diagnostic

模型冻结后，在 release6 seed identity 上运行 20 个 curve/mixed path-suite 场景；这与主 safety suite 使用不同场景定义，因此只作为 post-freeze diagnostic，不作为第二个独立 release set。

在相同 longitudinal supervisor 下：

| | Base | v2.7 expert-distilled |
|---|---:|---:|
| success | 5% | **10%** |
| collision | 0% | 0% |
| off-road | 95% | **80%** |
| timeout | 0% | 10% |
| route completion | 25.97% | **36.35%** |
| mean lane error | 0.782 m | **0.681 m** |

这个结果支持“privileged lateral teacher 提供了方向性的 path-following 信号”，但 **80% off-road 明确表示横向能力仍未通过**。

另一个诊断：直接让 privileged IDM steering 参与真实 MetaDrive 控制，curve/mixed 场景可以显著抑制 off-road，但常出现 timeout；full IDM + longitudinal supervisor 在 dense 场景也大量 timeout。这说明当前 benchmark 同时包含**横向路径能力与拥堵交互/效率**两个独立瓶颈，不能仅通过 imitation epoch 数解决。

## 6. v2.7 结论

v2.7 的主要成果不是“得到安全新模型”，而是把横向瓶颈做成可训练、可隔离、可冻结验证的研究问题：

```text
MetaDrive privileged lateral expert
          ↓
real expert_train correction data
          ↓
student policy distillation
          ↓
validation2 selection + promotion gate
          ↓
model hash freeze
          ↓
release6 one-time evaluation
          ↓
path-specific secondary diagnosis
```

最终模型虽然按 validation2 gate 进入 registry，但 registry 状态明确标记为 **release6 not confirmed / not safety certified**。release6 已消费，后续不得根据 12000–12039 继续调参并再把它当未见测试集。

## 7. 下一研究阶段

1. 新建 `interaction_train` / `validation3` / `release7`，禁止复用 release6；
2. 多 source PPO training seeds（至少 3）验证 source-policy 随机性；
3. 单独训练 curve/path-following policy 或 hierarchical lateral controller，再与 longitudinal RL 融合；
4. 对 dense traffic 引入 progress/stagnation objective 和 merge/yield 专项场景，区分“安全停车”与“死锁”；
5. 所有新模型同时门控 success、collision、off-road、timeout、supervisor dependency，禁止失败类型迁移制造改进；
6. 继续 raw RGB / BEV teacher-student，但视觉结果必须单独证明闭环驾驶能力。

## Claim boundary

- MetaDrive IDM teacher 是 training-only privileged teacher；
- release6 40 场景不能支持安全认证、事故率估计或真实道路结论；
- selected model 在 release6 raw collision 为 80%，不属于安全驾驶策略；
- SafetySupervisor 不是 CBF/MPC 形式化安全证明；
- 项目仅用于仿真/研究，禁止用于实车控制。
