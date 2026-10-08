# RL AutoDrive Agent v4.2.0-dev

面向 **AI 仿真算法 / 自动驾驶强化学习** 岗位的研究型工程项目。项目在 CPU 环境中提供轻量 native 后端和真实 **MetaDrive 0.4.3** 后端，覆盖 PPO、SAC、CTDE MAPPO、语义 BEV、raw-RGB+motion、ScenarioEnv/Waymo 小样本、SafetySupervisor、privileged expert distillation、严格的 validation/release 隔离，以及 v3.1 新增的 **Learned Route Adapter**：通过三轮 online DAgger 将 policy-visible adaptive RouteAssist 的横向修正蒸馏为神经残差模块，在 raw adapter 运行时不再调用 pure-pursuit。

## v4.2.0-dev 当前主线

v4.2 是在 v4.1 正式 no-promotion 之后建立的 **authority-gated longitudinal residual** 开发分支。v4.1 已证明 mixed-support longitudinal learning 能改善高交互场景，但 always-on residual 会在无邻车/OOD 状态中过度刹车并破坏成熟标准能力。v4.2 因此冻结 v2.8 base + v3.2 learned lateral，新增纵向 residual 的结构/学习双门控。

开发筛选只复用了已经消费的 `validation15`，不构成正式 v4.2 证据。单开发 seed 12400 的结果为：G0 14/15 standard、9/12 interaction；GP 14/15、9/12；GL 14/15 且 1 collision、6/12；**GPL 15/15 standard、0 catastrophic，同时 interaction 7/12**。因此 GPL 被选为正式架构，G0 保留为 always-on baseline。详见 `docs/V42_AUTHORITY_GATING_DEV_SELECTION.md`。

正式训练已经预注册为 `G0 vs GPL × {13400,14400,15400}`，只允许 `authority_train=47000–47999`。fresh `validation16=48000–48199` 与 `release19=49000–49199` 仍保持未打开；开发 seed 12400 被 formal launcher 硬性禁止复用。formal validation 规则见 `docs/V42_VALIDATION16_PROTOCOL.md`。

当前真实 MetaDrive 回归：**354 passed / 1 skipped**；唯一 skip 是在 MetaDrive 已安装时无法执行“缺依赖行为”反向测试。



> **v3.2 边界**：新增 `CurveRouteAdapter`，在真实 MetaDrive curve/ramp DAgger 中采集 4,417 transitions。fresh release15 的 primary raw evaluation **不调用几何 RouteAssist，也不调用 SafetySupervisor**：v3.1→v3.2 为 **60.0%→93.3% success、13.3%→0% collision、26.7%→6.7% off-road**。paired success 为 5 改善 / 0 退步，exact McNemar p=0.0625；样本仅15场景且 base actor仍是单一训练谱系，不能宣称统计显著或安全认证。curve tracking 从0/3提升到2/3，但仍有1个off-road。

## v3.3.0-dev 当前开发状态

v3.3 不是已晋升 release，而是从 v3.2 的单帧 curve residual 继续推进的开发分支。当前已完成代码层面的 **Scene-Aware Temporal Adapter**：

- `state history -> temporal route GRU`，默认 6 帧；
- 6 个邻车由固定 flatten slot 改为 masked set attention，并保留 `--no-use-neighbors` 消融入口；
- 增加可选的 `safety-risk` auxiliary head，训练时可读取 SafetySupervisor intervention/severity 标签，**运行时绝不修改动作**；scene_train pilot 显示当前 R15 可能伤害 steering，因此默认 risk-loss 已改为 0，必须显式 opt-in；
- 公共 `LearnedRouteAdapterAgent` 已支持 v1/v2/v3 artifact，并在 episode reset 清空 temporal history；
- evaluator warm-up 后重新 reset，避免 repeated reset-frame 污染 recurrent history；
- DAgger dataset schema 扩展为 base/learner/teacher/executed/applied action、step、reward、supervisor label 与 failure label，为后续 online PPO / constrained RL 留接口；
- 新建 `scene_train=37000–37999`、`validation12=38000–38199`、`release16=39000–39199`。本轮真实 evaluator smoke 使用了 validation12 seed 38000，因此 **validation12 已永久标记 engineering-consumed**；正式 candidate selection 已迁移到 fresh `validation13=40000–40199`。release16 仍未打开。

用户提供的 MetaDrive 0.4.3 assets 已接入隔离 runtime。当前全量回归为 **323 passed / 1 skipped**，其中 9 个真实 MetaDrive marker tests 全部通过；另完成 headless reset/step、v3.3 双 episode temporal reset、protected mini DAgger、raw learner deployment 和 A/B/C/D 短训练 smoke。上述 smoke 不构成 v3.3 性能结论，不要把 v3.2 release15 数字当成 v3.3 性能。详细实现与下一实验矩阵见 `docs/SCENE_AWARE_TEMPORAL_ADAPTER_v3.3.0-dev.md`。

### v3.3 训练入口

```bash
python scripts/train_scene_aware_adapter.py \
  --base-model models/metadrive_ppo_state_capability_distilled_v28.pt \
  --output experiments/v3_3/scene_adapter \
  --split scene_train --iterations 3 --seeds-per-stage 2 \
  --history-len 6 --hidden-size 128 --neighbor-heads 4 \
  --collection-safety protected

# validation12 已被 engineering smoke 消费；正式 selection 只允许使用 fresh validation13
# release16 必须在 candidate/freeze 完成后再首次打开
python scripts/evaluate_learned_route_suite.py \
  --model models/metadrive_ppo_state_capability_distilled_v28.pt \
  --steering-adapter experiments/v3_3/scene_adapter/scene_aware_temporal_adapter.pt \
  --split validation13 --suite capability --seeds-per-stage 5 \
  --output reports/v3_3/validation13_candidate
```

A/B/C/D paired representation ablation（默认 4 个结构 × 3 个独立 adapter initialization seeds；四组统一 `risk-loss-weight=0`，selection 仅使用 `validation13`）：

```bash
python scripts/run_v33_representation_ablation.py \
  --base-model models/metadrive_ppo_state_capability_distilled_v28.pt \
  --output experiments/v3_3/representation_ablation
```

SafetySupervisor risk auxiliary 不再与 neighbor representation 混在同一消融里，使用独立 R0/R15 launcher：

```bash
python scripts/run_v33_risk_aux_ablation.py \
  --base-model models/metadrive_ppo_state_capability_distilled_v28.pt \
  --output experiments/v3_3/risk_aux_ablation
```

两个 launcher 都不暴露 `release16` 参数，并支持 **job-aware matrix resume**：fresh trial 正常启动、partial trial 才注入 `--resume`、completed trial 直接跳过；矩阵自身还保存 plan fingerprint。底层 `train_scene_aware_adapter.py --resume` 继续要求原始 `run_config.json` 的完整 fingerprint 匹配，防止换超参后写入同一实验目录。

## v3.2.0 关键实测结果

v3.2 将 v3.1 的 residual+blend adapter 改为 **curve-aware direct neural residual**：route preview 进入 route-intent encoder，teacher correction 直接作为 steering residual 学习目标。fixed 0.5 pure-pursuit 只在训练数据采集阶段作为 teacher；release runtime 不存在规则 steering 接管。

| Fresh Release15 (36000–36014) | v3.1 raw | **v3.2 raw** |
|---|---:|---:|
| Success | 60.0% | **93.3%** |
| Collision | 13.3% | **0%** |
| Off-road | 26.7% | **6.7%** |
| Catastrophic | 40.0% | **6.7%** |
| Route completion | 73.45% | **92.41%** |
| Lane error | 0.435 m | **0.196 m** |

能力分解：curve 2/3、ramp 3/3、bottleneck 3/3、intersection 3/3、dense 3/3。完整证据见 `docs/CURVE_ROUTE_ADAPTER_v3.2.0.md`、`reports/v3_2/comparison.json` 与 `reports/v3_2/release15_freeze.json`。validation11/release15 已消费。

### v3.2 CurveRouteAdapter 复现

```bash
python scripts/train_curve_route_adapter.py \
  --base-model models/metadrive_ppo_state_capability_distilled_v28.pt \
  --output experiments/my_curve_adapter \
  --split curve_train --iterations 3

# release15 已消费，仅用于历史复核，不得再次用于调参
python scripts/evaluate_learned_route_suite.py \
  --model models/metadrive_ppo_state_capability_distilled_v28.pt \
  --steering-adapter models/metadrive_curve_route_adapter_v32.pt \
  --split release15 --suite capability --seeds-per-stage 3 \
  --output reports/my_release15_recheck
```

> **边界**：v3.0 **仍没有晋升新的 learned checkpoint**。发布系统继续使用冻结的 v2.8 神经策略；区别是 RouteAssist 从 v2.9 的固定 100% 几何 steering 替换，改成最大 blend=0.5 的自适应 shared control。fresh release13 15 场景中 adaptive system 达到 **80.0% success / 0% collision / 20.0% off-road**，与固定 RouteAssist 的 success/catastrophic 相同，但平均几何 steering blend 从 **100% 降至 11.4%**，对应 **88.6% 平均 learned steering authority**。它仍是 hybrid neural+rule system：RouteAssist 在约 **76.5%** 时间步产生非零小修正，SafetySupervisor 在约 **14.1%** 时间步修改纵向控制；这些指标绝不能归因于 raw RL policy，也不是安全认证。

> **边界（v3.1）**：Learned Route Adapter 的训练 teacher 是手写 pure-pursuit RouteAssist，但 release14 的 **raw adapter** 运行时没有调用几何 RouteAssist。fresh release14 15场景中 raw base → learned adapter 为 **33.3%→60.0% success、26.7%→6.7% collision、40.0%→33.3% off-road**；paired success 为4改善/0退步，exact McNemar p=0.125，样本量不足以宣称统计显著。curve tracking仍为0% success/100% off-road。带SafetySupervisor的66.7% success属于learned-lateral + rule-longitudinal混合系统。

## v3.1.0 关键实测结果

v3.1保持冻结的v2.8 base actor不变，新增`LearnedRouteAdapter`（state + base action → residual steering + learned blend）。真实MetaDrive `adapter_train` 31000–31014上进行三轮DAgger，teacher执行比例依次1.0/0.5/0.2，共采集**3,634 transitions / 15 unique seeds**。

Validation10（32000–32014）仅用于选择：raw base 33.3% success / 66.7% catastrophic，learned adapter raw **46.7% / 53.3%**，满足预声明`+10pp success + -10pp catastrophic` gate。随后在release14打开前冻结adapter/model/config/selection/core source SHA。

Release14（33000–33014）首次打开后的结果：

| Release14 | Raw base | **Learned adapter raw** | Adapter + SafetySupervisor | Adaptive rule reference |
|---|---:|---:|---:|---:|
| Success | 33.3% | **60.0%** | 66.7% | 80.0% |
| Collision | 26.7% | **6.7%** | **0%** | 0% |
| Off-road | 40.0% | **33.3%** | 33.3% | **20.0%** |
| Route completion | 58.3% | **68.4%** | 72.3% | 87.2% |

能力分解仍显示明显短板：ramp merge达到3/3、dense progress 2/3，但**curve tracking仍0/3并全部off-road**。因此v3.1证明的是“部分规则横向能力成功迁移到learned residual”，不是“learned policy已经解决路线跟踪”。完整证据见`docs/LEARNED_ROUTE_ADAPTER_v3.1.0.md`、`reports/v3_1/comparison.json`和`models/route_adapter_registry.json`。validation10/release14现已消费。

### v3.1 Learned Route Adapter 复现

```bash
python scripts/train_learned_route_adapter.py \
  --base-model models/metadrive_ppo_state_capability_distilled_v28.pt \
  --output experiments/my_route_adapter \
  --iterations 3 --seeds-per-stage 1 --mix-betas 1.0,0.5,0.2

# 精确复现实验 suite（validation10/release14 已经消费，仅用于复核，不得再用于调参）
python scripts/evaluate_learned_route_suite.py \
  --model models/metadrive_ppo_state_capability_distilled_v28.pt \
  --steering-adapter models/metadrive_learned_route_adapter_v31.pt \
  --split release14 --suite capability --seeds-per-stage 3 \
  --output reports/my_adapter_release14
```

## v3.0.0 关键实测结果

v3.0 不更换神经 checkpoint，而是把 v2.9 接近全时接管 steering 的 pure-pursuit RouteAssist 改为 **adaptive shared control**。need score 只使用 policy-visible 的 route curvature、heading error 和 lane offset；几何 steering blend 在 0～0.5 内按风险/跟踪需求连续变化。

Validation9（29000–29014）只用于系统参数选择：fixed route+safety 为 86.7% success / 13.3% off-road；adaptive max-blend=0.5 为 **93.3% success / 6.7% off-road**，平均 route blend **13.7%**。在任何 release13 请求产生前，`reports/v3_0/release13_freeze.json` 冻结了模型、配置、核心控制/评估源码及 selection SHA。

随后首次打开 release13（30000–30014，15个 fresh MetaDrive capability 场景）：

| Release13 | Raw v2.8 | Fixed v2.9 semantics | **Adaptive v3.0** |
|---|---:|---:|---:|
| Success | 26.7% | 80.0% | **80.0%** |
| Collision | 6.7% | 0% | **0%** |
| Off-road | 66.7% | 20.0% | **20.0%** |
| Route completion | 47.6% | **92.9%** | 84.5% |
| Mean geometric steering blend | 0% | 100% | **11.4%** |
| Mean learned steering authority | 100% | 0% | **88.6%** |

Fixed→adaptive 的 paired success 为 **1 改善 / 1 退步 / 11 双成功 / 2 双失败，exact McNemar p=1.0**；因此不能宣称 aggregate success 优于固定规则。v3.0 的实质贡献是：在这组有限 holdout 上保持同样 success/catastrophic 的同时，显著降低几何转向控制权。代价是 route completion 从 92.9% 降到 84.5%，且 curve stage 从 2/3 success 降到 1/3；adaptive 仍在约 76.5% 的时间步产生非零修正。

`models/registry.json` 继续只登记 learned checkpoints；v3.0 shared-control system 记录在 `models/system_registry.json`。完整报告见 `docs/ADAPTIVE_SHARED_CONTROL_v3.0.0.md` 和 `reports/v3_0/comparison.json`。validation9/release13 已消费，后续调参必须使用新 validation/release 域。

## v2.9.0 关键实测结果

### Route schema v4 与 learned lateral 负实验

新增 route observation schema v4：路线预览按**路径弧长**向前采样，不再使用旧 schema3 的 ego-frame x 排序。schema3 保持默认兼容，已有模型不受破坏。真实 MetaDrive 中运行了 schema4 pure-expert、steering-weighted、DAgger 以及 closed-form lateral-head ridge 多条路线。它们可以显著降低离线 steering MSE，但在 validation8 的 raw 闭环 curve/ramp 上仍出现高比例出界，因此 **没有任何 v2.9 learned lateral candidate 进入 curated registry**。

### 透明 hybrid system

v2.9 把规则控制明确放在系统边界中，而不是藏进神经策略：

```text
frozen v2.8 neural actor
       │
       ├─ steering ─────→ pure-pursuit RouteAssist
       │                  （只读取 policy-visible route preview）
       │
       └─ longitudinal ─→ SafetySupervisor（仅在风险时纠正）
                              ↓
                         MetaDrive 0.4.3
```

Validation8（26000-26014，15个场景）：

| System | Success | Collision | Off-road | Route completion | Lane error |
|---|---:|---:|---:|---:|---:|
| raw v2.8 | 26.7% | 13.3% | 60.0% | 49.7% | 0.566 m |
| + RouteAssist | 66.7% | 26.7% | 6.7% | 83.2% | 0.063 m |
| **+ RouteAssist + SafetySupervisor** | **93.3%** | **0%** | **6.7%** | **97.3%** | **0.062 m** |

在任何 release12 评估之前，项目先写入 `reports/v2_9/release12_freeze.json`，冻结模型 SHA、系统配置、评估脚本和核心源文件 SHA。之后才第一次打开 release12（27000-27014，15个 fresh MetaDrive 场景）：

| Release12 | Raw v2.8 | v2.9 hybrid |
|---|---:|---:|
| Success | 33.3% | **93.3%** |
| Collision | 13.3% | **0%** |
| Off-road | 53.3% | **6.7%** |
| Collision OR off-road | 66.7% | **6.7%** |
| Route completion | 53.2% | **96.7%** |
| Lane error | 0.599 m | **0.071 m** |

配对 success 为 **10 改善 / 1 退步 / 4 双方成功**，描述性 exact McNemar **p=0.0117**。五个 capability stage 中 curve、ramp、bottleneck、dense 均达到 3/3 success；intersection 为 2/3，剩余失败是 off-road。

但解释边界同样重要：release12 hybrid 的 RouteAssist intervention fraction 是 **99.82%**，说明横向控制几乎由几何控制器主导；纵向 SafetySupervisor intervention fraction 为 **15.40%**。因此该结果证明的是**透明混合控制系统在有限仿真场景上的改善**，不是 raw neural policy 已经学会 curve/ramp。完整报告见 `docs/HYBRID_ROUTE_CONTROL_v2.9.0.md` 和 `reports/v2_9/comparison.json`。release12 已永久消费，后续调参必须新建 validation9/release13。

下面命令仅用于**复核已经消费的 validation8 历史证据**，不得再据此调参；新实验必须先在 `src/scenarios/generator.py` 预留全新的 validation9/release13 或更晚 seed 域，再复用相同透明控制开关：

```bash
python scripts/evaluate_safety_suite.py \
  --model models/metadrive_ppo_state_capability_distilled_v28.pt \
  --split validation8 --suite capability --seeds-per-stage 3 \
  --route-assist --supervised --route-blend 1.0 \
  --ttc-threshold 1.8 --risk-threshold 0.68 \
  --output reports/my_hybrid_diagnostic
```

`models/registry.json` 只登记 learned checkpoints；`models/system_registry.json` 单独登记 v2.9 hybrid system，防止把规则系统指标混入神经模型指标。

## v2.8.0 关键实测结果

v2.8针对v2.7暴露的横向/path与交互能力瓶颈，构建五阶段 MetaDrive capability suite：`curve_tracking / ramp_merge / bottleneck_yield / intersection_yield / dense_progress`。开发中 validation3-6 分别暴露 anchored、balanced、component-wise 和 bounded imitation 的负结果，每次消费后都迁移到新的 validation domain。最终候选只在 **validation7 22000-22014** 上选模。

最终 pure-expert 行为克隆使用 `MetaDrive IDMPolicy` 作为**训练期特权全动作teacher**，在 `capability_train` 的 **13500-13529 共30个独立seed** 上收集 **9,000 transitions**；采集时100%执行expert action，部署时只运行student neural actor。

Validation7（15场景）：

| Policy | Success | Collision | Off-road | Catastrophic |
|---|---:|---:|---:|---:|
| v2.7 source | 6.7% | 40.0% | 53.3% | 93.3% |
| **v2.8 pure-expert BC** | **26.7%** | **20.0%** | 53.3% | **73.3%** |

候选在release11打开前通过预声明工程promotion gate。随后才首次运行 **release11 23000-23024 共25场景**：

| Policy | Success | Collision | Off-road | Catastrophic | Route completion |
|---|---:|---:|---:|---:|---:|
| v2.7 source | 4.0% | 32.0% | 64.0% | 96.0% | 46.9% |
| **v2.8 capability BC** | **32.0%** | **12.0%** | **56.0%** | **68.0%** | **56.2%** |

paired success 为 **8改善 / 1退步，exact McNemar p=0.0391**。这是有限MetaDrive场景的描述性研究证据，不是跨训练seed或真实道路安全证明。

能力分解显示提升并不统一：`bottleneck_yield / intersection_yield / dense_progress` 分别达到40% / 60% / 60% success；但 `curve_tracking` 仍 0% success / 100% off-road，`ramp_merge` 从source 20% success降到0%。因此v2.8可以宣称**aggregate capability improvement**，不能宣称“所有自动驾驶能力均提升”或“策略安全”。完整报告见 `docs/CAPABILITY_SPECIALIZATION_v2.8.0.md`。

## v2.7.0 关键实测结果

v2.7针对v2.6暴露的横向/path bottleneck，在真实MetaDrive中使用内置 `IDMPolicy` 作为**训练期特权lateral teacher**。正式 steering-only 与 balanced 两条路线共采集 **7,014 expert transitions / 32个互斥expert_train seeds**；部署时 raw student 仍只读取项目state observation。另新增curve/mixed path suite、source-action/feature anchored distillation负实验，以及新的 `expert_train / validation2 / release6` 防泄漏seed域。

Validation2固定20场景（11000–11019）：

| Candidate | Success | Collision | Off-road | Catastrophic |
|---|---:|---:|---:|---:|
| v2.5 source | 15% | 85% | 0% | 85% |
| steering-only expert | 10% | 85% | 5% | 90% |
| **balanced expert** | **20%** | **75%** | 5% | **80%** |
| anchored medium | 20% | 80% | **0%** | 80% |

`balanced` 相对source success +5pp、collision -10pp、off-road +5pp，恰好通过预声明 engineering promotion gate。选择报告和模型哈希随后写入 `reports/v2_7/FROZEN_MODEL.json`，**release6在此之后才第一次打开**。

冻结release6 12000–12039共40场景：

| System | Success | Collision | Off-road | Timeout | Safety介入 |
|---|---:|---:|---:|---:|---:|
| v2.5 source raw | 15.0% | 77.5% | 7.5% | 0% | 0% |
| v2.7 expert-distilled raw | **17.5%** | 80.0% | **2.5%** | 0% | 0% |
| v2.5 source + longitudinal supervisor | 25.0% | **0%** | 50.0% | 25.0% | 51.47% |
| v2.7 expert-distilled + supervisor | 22.5% | 7.5% | **32.5%** | 37.5% | **25.36%** |

Raw paired success只有 **2改善 / 1退步，exact McNemar p=1.0**；collision-or-off-road仅85%→82.5%。因此 release6 **没有确认validation2 promotion对应的稳健安全收益**。同一supervisor下规则介入约减半，但系统success反而略降，不能把“规则依赖下降”包装成“系统更安全”。

Secondary path diagnostic（20个curve/mixed场景）显示selected的off-road 95%→80%、route completion 25.97%→36.35%、mean lane error 0.782→0.681m，说明teacher提供了方向性的横向信号，但 **80% off-road仍然明确不通过能力目标**。完整冻结报告见 `docs/SAFETY_PATH_EXPERT_v2.7.0.md` 和 `reports/v2_7/comparison.json`。

## v2.6.0 关键实测结果

v2.6 使用互斥的 safety-training seeds 做三条真实 MetaDrive correction-learning 路线：strict longitudinal DAgger、balanced encoder+actor DAgger，以及仅用于训练标签的 lane teacher。另从 strict DAgger warm-start 运行 3×1024-step staged PPO。候选选择只使用 validation **1000–1019** 共20场景；选择规则和 promotion gate 固化在 `src/evaluation/model_selection.py`。

| Validation候选 | 成功率 | 碰撞率 | 出界率 | 碰撞或出界 | 是否晋升 |
|---|---:|---:|---:|---:|---:|
| v2.5 base | 25% | 70% | 5% | 75% | baseline |
| strict longitudinal DAgger | 25% | 45% | 30% | 75% | 否 |
| **balanced DAgger（排名第一）** | **25%** | **45%** | **25%** | **70%** | **否** |
| lane-teacher DAgger | 25% | 60% | 15% | 75% | 否 |
| staged PPO | 25% | 65% | 10% | 75% | 否 |

预声明 gate 要求至少 `+5pp success` 或 `-10pp catastrophic`，且 collision 不恶化、off-road 增幅不超过5pp。`dagger_balanced` 相对 base 只有 **-5pp catastrophic**，同时 **off-road +20pp**，因此在 release5 打开前就已确定“不晋升”。

模型冻结后首次打开 release5 的前40个场景（**9000–9039**）：

| release5 系统 | 成功率 | 碰撞率 | 出界率 | Timeout | Safety介入时间 |
|---|---:|---:|---:|---:|---:|
| v2.5 base raw | 22.5% | 75% | 2.5% | 0% | 0% |
| v2.6 selected raw | **25%** | **65%** | **10%** | 0% | 0% |
| v2.5 base + longitudinal supervisor | 25% | 2.5% | 50% | 22.5% | 51.66% |
| v2.6 selected + longitudinal supervisor | 25% | 0% | 50% | 25% | **23.61%** |

raw candidate 的 collision 比 base 低10pp，但 off-road 高7.5pp，`collision OR off-road` 仅从77.5%降到75%；配对 success 仅 **1改善 / 0退步，exact McNemar p=1.0**。supervisor 介入时间相对下降约54.3%，但监督系统成功率仍只有25%，说明当前主要瓶颈已经从纯纵向碰撞转向**弯道/混合地图的横向与路径跟踪失败**。release5 已经被查看，后续任何调参必须使用新的未触碰 release split。

详细方法与负结果见 `docs/SAFETY_CURRICULUM_v2.6.0.md`，冻结对照见 `reports/v2_6/comparison.json`。

## v2.5.0 关键实测结果

固定32,768-step MetaDrive state PPO作为 source。SafetySupervisor 在 12 个独立 safety-training seeds 上真实执行 actor 的纵向安全修正，共采集 **1,419 transitions / 733 interventions**。候选超参数只使用 validation 1000–1009；模型冻结后才首次评估 **release4 7000–7039 共40场景**：

| 系统 | 成功率 | 碰撞率 | 出界率 | 平均速度 | 安全层介入时间占比 |
|---|---:|---:|---:|---:|---:|
| Base raw PPO | 42.5% | 57.5% | 0% | 10.58 m/s | 0% |
| **Safety-distilled raw PPO** | **60.0%** | **40.0%** | 0% | 8.81 m/s | 0% |
| Base PPO + SafetySupervisor | 100% | 0% | 0% | 8.95 m/s | 47.22% |
| **Distilled PPO + SafetySupervisor** | **100%** | **0%** | 0% | 8.17 m/s | **27.11%** |

蒸馏后 supervisor 介入时间相对下降约 **42.6%**，平均纵向修正幅度下降约 **62.5%**；40场景中38个介入比例下降。但 raw collision 仍为40%，且paired raw outcome仅10个场景改善、3个退步（描述性exact McNemar p≈0.092）。因此结论是**规则依赖下降、raw策略有所改善，但仍远未达到安全驾驶要求**。

### Raw RGB + motion

`rgb_ego` 使用真实 MetaDrive `RGBCamera` 像素以及显式 ego/navigation motion；不是 semantic BEV，也不是 camera-only policy。

- PPO 真实 RGB 训练：**384 world steps**，CNN 参数 L1 变化约 **90.82**，模型和 TensorBoard 事件已保存；40-step smoke 中路线完成度约 4.72%，平均速度约 0.05 m/s，**不具备驾驶能力**。
- 特权 state teacher → RGB+motion student 行为克隆：从 6 个训练 seed 收集 48 个真实像素样本，MSE **0.37147 → 0.000150**；40-step smoke 路线完成度约 20.83%，平均速度约 4.93 m/s，仍以 timeout 结束，**只证明跨观测蒸馏链路工作**。

## 已实测的真实 MetaDrive 能力

- MetaDrive 0.4.3 源码、assets、离线 wheelhouse 安装及真实 reset/step；
- 自研 PPO state 累计训练 32,768 world steps；
- 两学习车辆、同一物理世界的 CTDE MAPPO 8,192 world steps；
- semantic `bev_ego×4` CNN PPO 4,096 world steps（链路通过，但 10/10 测试出界）；
- 官方 observation/reward + Stable-Baselines3 PPO 参考训练；
- 3 个 assets 自带 Waymo/ScenarioNet 示例的闭环 ego 评估；
- MetaDrive `RGBCamera` 在 CPU + Xvfb 下真实采样；
- fresh-process 分段训练 + checkpoint 恢复，降低长 Panda3D 进程异常带来的损失。

## 项目结构

```text
configs/                 环境、算法、奖励、场景与安全/RGB配置
src/
  agents/                PPO / SAC / MAPPO / transfer / RGB distillation
  envs/                  native / MetaDrive、reward v2-v4、observation
  safety/                SafetySupervisor + intervention-aware actor distillation
  experts/               MetaDrive privileged training teacher + anchored distillation
  perception/            semantic BEV、RGB CNN、motion fusion、历史帧
  multi_agent/           MARL 环境与策略接口
  datasets/              轨迹数据检查、转换与回放
  evaluation/            闭环评估、置信区间、舒适性、失败轨迹
scripts/                 训练、评估、安全/RGB蒸馏、MetaDrive分段训练、发布工具
models/                  精选模型与 SHA-256 registry
experiments/             正式训练产物
reports/                 逐回合评估、失败轨迹和发布证据
tests/                   单元、集成及真实 MetaDrive 测试
docs/                    架构、API、验收矩阵、限制、开发台账
```

完整发行文件见 `docs/TREE.txt`。

## 快速开始：native

```bash
python -m pip install -r requirements.txt
python scripts/train.py --config configs/ppo.yaml --steps 4096 --output experiments/my_native_ppo
python scripts/evaluate.py --model experiments/my_native_ppo/model.pt --episodes 20 --split test --output reports/my_native_ppo
```

## MetaDrive 训练

准备与目标平台匹配的 MetaDrive 0.4.3 源码、assets 和 wheelhouse，部署方法见 `docs/METADRIVE_REAL_v2.3.0.md` 与 `docs/METADRIVE_OFFLINE_v2.2.0.md`。

项目 PPO：

```bash
python scripts/train_metadrive_segmented.py \
  --config configs/metadrive_ppo.yaml \
  --output experiments/metadrive_ppo_segmented \
  --total-steps 32768 --segment-steps 4096 --timeout 120
```

基础闭环评估：

```bash
python scripts/evaluate.py \
  --model models/metadrive_ppo_state_32768.pt \
  --episodes 20 --split test --output reports/metadrive_eval
```

## v2.5 SafetySupervisor correction distillation

真实 MetaDrive 安全修正采集与 actor 蒸馏：

```bash
python scripts/distill_metadrive_safety.py \
  --source models/metadrive_ppo_state_32768.pt \
  --collection-config configs/metadrive_supervised_eval.yaml \
  --output experiments/my_safety_distill \
  --iterations 1 --seeds-per-iteration 12 --max-steps-per-seed 300 \
  --epochs-per-iteration 8 --batch-size 256 \
  --learning-rate 2e-4 --intervention-weight 0.75 --steering-weight 0.35
```

发布模型 `models/metadrive_ppo_state_safety_distilled_v25.pt` 只把 actor/encoder 对 supervisor applied action 做 imitation；critic 与 exploration variance 不在蒸馏阶段训练。它不是完整 PPO resume checkpoint，也不是 safety-certified policy。最终 frozen release4 对照见 `docs/SAFETY_DISTILLATION_v2.5.0.md`。

## v2.4 安全监督对照

原始策略和监督系统必须分别跑、分别报告：

```bash
# raw policy
python scripts/evaluate.py \
  --model models/metadrive_ppo_state_32768.pt \
  --episodes 40 --split release3 \
  --output reports/raw_release3

# same frozen actor + independent longitudinal supervisor
python scripts/evaluate.py \
  --model models/metadrive_ppo_state_32768.pt \
  --episodes 40 --split release3 \
  --config configs/metadrive_supervised_eval.yaml \
  --output reports/supervised_release3
```

安全层不会改 steering，只能限制 longitudinal command；每一步是否介入、原因和风险均写入 `info` 与评估指标。它不是 CBF、安全证明或法规级功能。

## v2.4 Raw RGB 训练与蒸馏

真实 RGB PPO smoke：

```bash
python scripts/train_metadrive_segmented.py \
  --config configs/metadrive_ppo_rgb_ego.yaml \
  --output experiments/rgb_ppo \
  --total-steps 384 --segment-steps 128
```

使用冻结 state teacher 收集真实 MetaDrive RGB 并蒸馏 actor：

```bash
xvfb-run -a python scripts/distill_metadrive_rgb.py \
  --teacher models/metadrive_ppo_state_32768.pt \
  --config configs/metadrive_rgb_distill.yaml \
  --output experiments/rgb_distilled \
  --samples 48 --seed-count 6 --epochs 60
```

蒸馏数据由 teacher action 实际控制仿真产生；student 仅读取 RGB + ego/navigation motion。critic 和 exploration log-std 不在行为克隆阶段训练。

## v2.8 Capability specialization

训练期特权全动作teacher采集与student行为克隆：

```bash
python scripts/distill_metadrive_capability.py \
  --source models/metadrive_ppo_state_expert_distilled_v27.pt \
  --output experiments/my_capability_bc \
  --seed-offset 500 --seeds-per-stage 6 --max-steps-per-seed 320 \
  --epochs-per-stage 20 --steering-blend 1.0 --longitudinal-blend 1.0 --mode balanced
```

能力套件评估：

```bash
python scripts/evaluate_safety_suite.py \
  --model experiments/my_capability_bc/model.pt \
  --split validation7 --suite capability --seeds-per-stage 3 \
  --output reports/my_capability_validation
```

正式发布已消费 `release11`；以上示例只能用于新的未触碰validation/release域，不能拿release11继续调参。

## 测试与完整性

```bash
PYTHONPATH=/path/to/metadrive/source:$PWD python -m pytest -q
python scripts/verify_package.py
```

最终测试数、覆盖率和独立解压验证以发行旁的 `rl_autodrive_agent_v2.9.0_verification.json` 为准。

## 结果解释原则

- “SafetySupervisor 0 碰撞”是**组合系统**结果，不是 raw PPO 的安全能力；
- safety-distilled raw policy 在 release4 仍有 40% collision；不得用组合系统结果覆盖它；
- release4 仅有 40 个场景，不能支持安全认证或真实道路风险结论；
- `rgb_ego` 是 RGB + motion/navigation，不是 camera-only 端到端；
- 行为克隆 student 使用 privileged state teacher，不是独立 RGB RL；
- semantic BEV 使用仿真几何真值；
- 模型参数真实更新只证明训练链路发生，不证明策略收敛；
- 项目禁止用于实车控制。

v2.7 lateral expert见 `docs/SAFETY_PATH_EXPERT_v2.7.0.md`，v2.6 安全课程见 `docs/SAFETY_CURRICULUM_v2.6.0.md`，v2.5 安全蒸馏见 `docs/SAFETY_DISTILLATION_v2.5.0.md`，RGB历史见 `docs/SAFETY_RGB_v2.4.0.md`；当前验收状态见 `docs/acceptance_matrix.md`，限制见 `docs/KNOWN_LIMITATIONS.md`。
