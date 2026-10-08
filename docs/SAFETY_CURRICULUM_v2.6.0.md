# v2.6.0 — Safety curriculum, DAgger and frozen release-suite report

## 1. 目标与原则

v2.6.0 不以“再增加一个算法”为目标，而是针对 v2.5 暴露的三个问题做工程化研究：

1. learned policy 对纵向 SafetySupervisor 仍有较强依赖；
2. dense/curve/mixed 地图上的失败不再只是纵向碰撞，存在明显横向/路径跟踪问题；
3. 多次 MetaDrive/Panda3D 环境构造可能在 teardown 阶段卡住，需要把评估结果的提交与进程生命周期解耦。

本版本采用严格的数据域约束：`safety_train` 用于 correction collection，validation 1000–1199 只用于候选选择与 promotion gate；`release5` 9000–9199 在候选选择冻结后才允许打开。一旦 release5 被查看，它永久成为已消费测试域，不能用于后续调参。

## 2. 实质性实现

### 2.1 多阶段 Safety DAgger

三种真实 MetaDrive correction-learning 路线都从 v2.5 curated model
`models/metadrive_ppo_state_safety_distilled_v25.pt` 开始：

| 实验 | safety-training seeds | transition | intervention | 训练范围 |
|---|---|---:|---:|---|
| `dagger_longitudinal` | 8020–8037 | 2,981 | 568 (19.05%) | encoder冻结，仅longitudinal actor row |
| `dagger_balanced` | 8060–8077 | 2,854 | 612 (21.44%) | encoder + actor；critic/log_std冻结 |
| `dagger_lane` | 8100–8117 | 2,821 | 2,574 combined | encoder冻结；纵向supervisor + 训练期lane teacher |

其中 `dagger_lane` 的 combined intervention 中，SafetySupervisor intervention 为845次（29.95%），lane teacher intervention 为2,446次（86.71%）。lane teacher 只读取 policy-visible 的归一化车道偏移、heading error 与 route preview，不读取世界坐标；它仅用于训练标签，**运行时 SafetySupervisor 仍然不修改 steering**。

训练阶段顺序为：

```text
straight_moderate
      ↓
straight_dense
      ↓
curve_dense
```

各路线均保留采集 seeds、source SHA-256、标签修正幅度、参数变化、critic/log_std训练状态和最终模型哈希。

### 2.2 staged PPO 负实验

`ppo_curriculum_from_dagger` 从 strict DAgger warm-start 继续做真实 MetaDrive PPO：3 个 stage × 1,024 world steps，共 3,072 world steps，并用 v2.5 source 作为 reference-policy KL 约束。该路线没有在 validation 上形成可晋升改进，作为负结果保留。

### 2.3 可恢复 fresh-process 安全评估

`evaluate_safety_suite.py` 为每个 suite stage 使用新进程，避免长生命周期 Panda3D teardown 卡住整个评估。`evaluation.complete.json` 只在 JSON/Markdown/HTML/CSV/失败轨迹等结果全部持久化后写入，父进程以这个 commit marker 作为阶段完成条件，然后可主动回收 worker。

`--resume` 只复用同时满足以下条件的阶段：

- `evaluation.complete.json` 存在且状态为 complete；
- stage request fingerprint 一致；
- model path/SHA、supervisor参数和suite配置一致；
- report 中的 seeds/env 与 request 完全一致。

残缺或fingerprint不匹配的阶段会被删除并重跑，不会把半写文件当作成功结果。

### 2.4 泄漏防护候选选择与 promotion gate

候选选择代码位于 `src/evaluation/model_selection.py`。它硬拒绝 validation 范围 1000–1199 以外的 seed，并要求所有候选使用完全相同的 suite。

排序规则按以下顺序固定：

1. 最大 success；
2. 最小 `collision OR off-road`；
3. 最小 collision；
4. 最小 timeout；
5. 最大 route completion；
6. name 仅作最终确定性 tie-break。

“排名第一”不等于“进入 curated registry”。预声明 promotion gate 相对 base 要求：

- `success gain >= 5pp` **或** `catastrophic reduction >= 10pp`；
- collision 不恶化；
- off-road 增幅不超过5pp。

该 gate 在 release5 打开前就完成并持久化。

## 3. Validation：候选选择与不晋升决定

Validation safety suite 使用 **1000–1019 共20个场景**，每个 stage 5 seeds：

| candidate | success | collision | off-road | collision OR off-road | timeout | mean route completion |
|---|---:|---:|---:|---:|---:|---:|
| v2.5 base | 25% | 70% | 5% | 75% | 0% | 42.78% |
| strict longitudinal DAgger | 25% | 45% | 30% | 75% | 0% | 47.59% |
| **balanced DAgger** | **25%** | **45%** | **25%** | **70%** | 5% | 45.24% |
| lane-teacher DAgger | 25% | 60% | 15% | 75% | 0% | 44.87% |
| staged PPO | 25% | 65% | 10% | 75% | 0% | 42.42% |

`dagger_balanced` 在固定排序中排名第一，但相对 base：

- success gain = **0pp**；
- catastrophic reduction = **5pp**；
- collision = **-25pp**；
- off-road = **+20pp**。

因此它同时触发 `no_material_success_or_catastrophic_improvement` 与 `offroad_regression`，**promotion = false**。curated `models/registry.json` 不新增 v2.6 模型。

完整机器记录：`reports/v2_6/candidate_selection.json`。

## 4. Release5：模型冻结后的首次诊断

在候选排序和 promotion 决定完成以后，首次打开 release5 的前40个场景 **9000–9039**。每个条件10 seeds：

- `straight_moderate` — `S`, density 0.18；
- `straight_dense` — `SSS`, density 0.35；
- `curve_dense` — `SCS`, density 0.30；
- `mixed_unseen` — `SCXTO`, density 0.22。

### 4.1 Raw policy

| policy | success | collision | off-road | timeout | catastrophic | route completion |
|---|---:|---:|---:|---:|---:|---:|
| v2.5 base | 22.5% | 75% | 2.5% | 0% | 77.5% | 43.94% |
| v2.6 selected | **25%** | **65%** | **10%** | 0% | **75%** | 47.71% |

差值：success +2.5pp，collision -10pp，off-road +7.5pp，`collision OR off-road` 仅 -2.5pp。

paired success：1改善 / 0退步 / 9双方成功 / 30双方失败；描述性 exact McNemar p=1.0。paired non-catastrophic outcome 同样只有1改善 / 0退步。40个固定场景不足以支持统计安全结论，且结果存在明确的 collision→off-road 失败类型迁移。

### 4.2 相同纵向 SafetySupervisor 下的系统表现

| system | success | collision | off-road | timeout | intervention time | mean longitudinal correction |
|---|---:|---:|---:|---:|---:|---:|
| v2.5 base + supervisor | 25% | 2.5% | 50% | 22.5% | 51.66% | 0.2774 |
| v2.6 selected + supervisor | 25% | 0% | 50% | 25% | **23.61%** | **0.0277** |

候选使 supervisor 介入时间相对下降约 **54.3%**，平均纵向修正下降约 **90.0%**；40场景中37个场景介入更少。但成功率没有提高：

- `straight_moderate`: 10/10 success；
- `straight_dense`: 10/10 timeout；
- `curve_dense`: 10/10 off-road；
- `mixed_unseen`: 10/10 off-road。

这说明当前 supervisor 对纵向碰撞过滤有效，但**没有解决 lateral/navigation failure**。因此“0 collision”不能作为“系统安全完成”的替代指标。

完整冻结比较：`reports/v2_6/comparison.json` 与 `reports/v2_6/comparison.md`。

## 5. 结论

v2.6.0 的主要成果是把安全改进过程从“看一个碰撞率”推进为可审计的训练/选择/冻结测试流程：

```text
safety_train correction data
        ↓
multiple DAgger / staged PPO candidates
        ↓
validation-only deterministic ranking
        ↓
predeclared promotion gate
        ↓
model freeze
        ↓
release5 one-time diagnostic
```

结果并不支持晋升新策略。最可信的工程结论反而是：**继续只学习纵向 supervisor 修正会出现边际收益，当前主要瓶颈是 dense/curve/mixed 场景的横向路径控制与低速死锁。** lane teacher 的第一版也没有在 validation 上解决这个问题。

下一版本如果继续研究，必须：

1. 把 release5 视为永久已消费测试集；
2. 新建独立 release6；
3. 在训练域内先建立 curve/merge/yield 的路径跟踪与交互课程；
4. 使用多个独立 source PPO training seeds 验证改进是否跨训练随机性成立；
5. 把 success、collision、off-road、timeout 与 supervisor dependency 同时作为门槛，禁止通过失败类型迁移制造“改进”。

## 6. Claim boundary

- v2.6 没有新模型进入 curated registry；
- release5 结果不能用于未来回选 v2.6 候选；
- longitudinal SafetySupervisor 不是 CBF/MPC 安全证明；
- lane teacher 是训练期启发式 teacher，不是认证控制器；
- 40个 release5 场景不能支持安全认证、真实道路事故率或统计确定的安全提升声明；
- 项目禁止用于实车控制。
