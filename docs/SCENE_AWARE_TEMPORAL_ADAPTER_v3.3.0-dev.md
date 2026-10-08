# v3.3.0-dev Scene-Aware Temporal Adapter

## 1. 目的

v3.2 的 `CurveRouteAdapter` 已证明 fixed pure-pursuit 的横向能力可以通过 DAgger 转移到 learned residual，并在 release15 的 raw runtime 中关闭 RouteAssist/SafetySupervisor 后仍保持 14/15 success。但它仍是单帧模型，且 steering adapter 完全忽略 `state[20:]` 的邻车表示。

v3.3 的目标不是扩大规则权重，而是验证两个表示假设：

1. 短时 route history 能否减少 curve/merge 中的瞬态与 route-preview 抖动；
2. 把邻车从固定距离排序 slot 改成 set representation，能否为后续 interaction-aware policy 与 safety-cost learning 提供稳定表示。

本开发版本不包含新的 release 性能结论。

## 2. 架构

```text
policy-visible state history [T, 20+7N]
        │
        ├── current ego[0:12] ── MLP ───────────────┐
        │                                            │
        ├── route[12:20] + angle/delta ─ MLP ─ GRU ─┤
        │                                            ├─ fusion ─ steering residual
        └── current neighbor tokens[N,7] ─ attention ┤
                                                     └─ diagnostic risk logit
base actor action[2] ─ action encoder ───────────────┘
```

默认配置：history=6、hidden=128、4-head neighbor attention、max steering correction=0.18。

### 运行时边界

- 输入仍然只是 actor 可见 state + frozen base action；
- 不读取 MetaDrive map geometry 或 privileged simulator state；
- 不调用 pure-pursuit；
- raw runtime 不调用 SafetySupervisor；
- risk head 只输出 diagnostic probability，不改变 steering/longitudinal action。

## 3. Risk auxiliary head：可选训练信号，不与表示消融混在一起

当前 lateral teacher 是 pure-pursuit，本身不读取邻车。如果只把 neighbor attention 接入 steering head，邻车分支没有直接监督目标，很容易退化成无用特征或偶然相关性。

因此 protected collection 会记录 `safety_intervention` 与 `safety_severity`。v3.3 可以用它们训练共享表示上的 auxiliary risk head，使 neighbor/ego interaction 获得一个真实训练信号；该 head 在部署时不接管控制。

但 auxiliary supervision 本身也是一个算法变量。为避免把“neighbor attention 的收益”和“额外 SafetySupervisor 标签的收益”混为一谈，A/B/C/D representation ablation **统一设置 risk-loss weight=0**。风险辅助监督另做 `R0 vs R15` 的独立消融，固定 full D representation，只改变 risk-loss weight。

如果 collection 为 `raw` 且整个数据集没有任何非零 risk label，训练器会自动把 effective risk-loss weight 置为 0，避免用“全安全”伪标签污染共享表示。

## 4. Temporal correctness

`LearnedRouteAdapterAgent` 持有最多 `history_len` 个 state；episode `reset()` 会清空 history。episode 开始不足 T 帧时，用第一帧左填充，而不是填全零状态。

旧 evaluator 原本会在 reset 后对同一个 observation 调 `predict()` 三次 warm-up。对 feed-forward v3.2 没影响，但会给 recurrent model 伪造 3 帧历史。v3.3 已修改两个 evaluator：warm-up 后再次调用 agent.reset()。

## 5. Neighbor representation

当前 observation 按距离排序邻车，这个顺序在车辆距离交叉时会突变。v3.3 不给 neighbor token 加 positional embedding，并用 masked multi-head attention 汇聚；因此对邻车 slot permutation 保持不变。

始终追加一个 learned null-neighbor token，防止“所有邻车 mask=false”时 attention softmax 产生 NaN，并允许网络显式选择 no-interaction context。

## 6. DAgger dataset schema

v3.3 collection 保存：

- `state`
- `base_action`
- `learner_action`
- `teacher_action`
- `teacher_steering`
- `teacher_correction`
- `executed_action`（进入 env 前的 student/teacher mix）
- `applied_action`（经过可选 SafetySupervisor 后实际执行）
- `seed`, `step`, `reward`
- `safety_intervention`, `safety_severity`
- `collision`, `out_of_road`, `success`

这让同一份数据以后可以直接扩展到 intervention prediction、cost critic、offline diagnostics 和 expert-guided online PPO，而不必重新设计日志协议。

## 7. Fresh domains

```text
scene_train   37000–37999
validation12  38000–38199   # engineering-consumed; 不再用于 candidate selection
release16     39000–39199
validation13  40000–40199   # fresh candidate-selection holdout
```

release15 只保留 v3.2 历史证据。为了验证 temporal evaluator 的 fresh-process/commit 路径，本开发阶段使用过 `validation12` 的 seed 38000；随后又用同 seed 做了 v3.2 sanity baseline。因此 validation12 已永久降级为 engineering-consumed domain，不能再承担正式选模。candidate selection 迁移到全新的 `validation13`。release16 仍未打开；candidate 必须先在 validation13 上按预注册协议选择并冻结 base/adapter/config/core-source hash，之后才允许首次打开 release16。

## 8. 第一轮消融

同一 base actor、同一 DAgger seeds、同一 teacher：

| ID | history | neighbors | risk aux | 目的 |
|---|---:|---:|---:|---|
| A | 1 | off | off | 接近 v3.2 direct residual control |
| B | 6 | off | off | 只测 temporal route |
| C | 1 | on | off | 只测 scene interaction representation |
| D | 6 | on | off | v3.3 full representation |

Risk auxiliary 单独做：

| ID | representation | risk-loss weight | 目的 |
|---|---|---:|---|
| R0 | D | 0.00 | 无辅助风险监督 |
| R15 | D | 0.15 | SafetySupervisor risk auxiliary |

两个消融都只允许使用 `scene_train` 训练，并按预注册方案在 `validation13` 做 selection；不要根据 release16 做任何结构或超参选择。

## 9. 当前验证

已经完成：

- identity initialization；
- temporal route history sensitivity；
- neighbor permutation invariance；
- null-neighbor finite attention；
- v3 artifact common loader/base SHA binding；
- history builder 不跨 seed/episode；
- agent reset 清空 temporal history；
- v3.2 regression；
- MetaDrive 0.4.3 + version-matched assets headless reset/64-step smoke；
- v2.8 frozen base + identity v3.3 adapter 两 episode closed-loop smoke，state=62D、history 1→6、episode reset 清零、所有 action/diagnostic finite；
- protected mini DAgger：curve/ramp 各 1 seed、200 transitions、15 次 SafetySupervisor intervention label，训练/保存/重载成功；
- trained mini adapter raw rollout：RouteAssist=OFF、SafetySupervisor=OFF、environment applied action 与 agent action 一致；
- A/B/C/D 四种 representation 均完成真实 MetaDrive 短训练 smoke；
- 正式 evaluator 在 validation12 seed 38000 完成 fresh-process commit smoke；该动作因此消费了 validation12，并触发 selection domain 迁移到 validation13；
- 全量 test regression：323 passed / 1 skipped（含 9 个真实 MetaDrive marker tests）。

上述 mini training / validation12 smoke 只验证运行与实验协议，不是 v3.3 candidate 性能证据。2-epoch D-mini 在 validation12 seed 38000 上仍 off-road；同一 seed 的历史 v3.2 adapter sanity baseline 成功，说明 mini checkpoint 明显欠训练，不应据此评价 v3.3 架构优劣。

随后又在 **scene_train-only** 条件下做了一个不打开 validation13 的二轮 DAgger pilot：A/B/C/D 使用相同 adapter initialization seed=3300、8 个 paired scenario seeds、`beta=1.0→0.2`、5 epoch、120-step cap。iteration 1 的 diagnostic steering MSE 为 A≈4.63e-3、B≈4.08e-3、C≈4.79e-3、D≈3.87e-3；四组均为 0 collision、1/4 off-road。D 在这个训练域 pilot 的 teacher-fit 最稳，但闭环 failure count 尚未拉开，不能当作泛化结论。

同设置的 R15 risk-aux pilot 将 D/R0 的 iteration-1 steering MSE 从约 3.87e-3 恶化到约 5.75e-3，off-road 仍为 1/4；同时 diagnostic risk target 极度偏向 0。基于这个开发域证据，`train_scene_aware_adapter.py` 的 **默认 risk-loss weight 已改为 0.0**，risk auxiliary 变成显式 opt-in，只通过 R0/R15 独立实验评估。

## 10. 下一步（进入 v3.3 实验）

1. 跑 A/B/C/D，各至少 3 个 adapter training seeds；`scripts/run_v33_representation_ablation.py` 默认使用 3300/4300/5300，并保持 scenario seeds paired；
2. 跑预注册的 R0/R15 risk-aux ablation，仍保持 paired scenario/training seeds；
3. 只在 `scene_train` 做训练和 failure mining；
4. 两套预注册实验统一在 fresh `validation13` 上做 candidate selection；
5. 若预声明 gate 通过，写 freeze manifest；
6. freeze 后首次跑 release16；
7. 若 temporal/full adapter 仍不能稳定消除 curve off-road，再进入 v4.0 expert-guided PPO，而不是继续调 pure-pursuit blend。


## 11. 可执行消融入口与复现约束

批量入口：

```bash
python scripts/run_v33_representation_ablation.py \
  --base-model models/metadrive_ppo_state_capability_distilled_v28.pt \
  --output experiments/v3_3/representation_ablation
```

该入口默认生成 A/B/C/D × 3 initialization seeds = 12 个 trial。所有 trial 共用 `scene_train` scenario seeds，四组统一关闭 risk auxiliary loss，并只在 fresh `validation13` 上做 candidate selection。脚本没有 release16 选项，避免 selection 阶段误触 release split。

风险辅助监督独立入口：

```bash
python scripts/run_v33_risk_aux_ablation.py \
  --base-model models/metadrive_ppo_state_capability_distilled_v28.pt \
  --output experiments/v3_3/risk_aux_ablation
```

该入口固定 D representation（history=6 + neighbor attention），默认比较 `R0=0.0` 与 `R15=0.15`，同样只使用 `scene_train + validation13`，不暴露 release16。

`train_scene_aware_adapter.py` 现在显式记录 `--training-seed`，并在 run 开始时写 `run_config.json` + SHA-256 fingerprint。`--resume` 时必须完整匹配原 run config；任何结构、数据、teacher、优化或 seed 参数变化都会拒绝续跑。


## 长矩阵恢复语义

A/B/C/D 与 R0/R15 launcher 现在采用 job-aware resume，而不是把 `--resume` 无条件传给所有 trial：

- fresh trial：即使矩阵级使用 `--resume`，也按新实验启动；
- partial trial：存在原始 `run_config.json` 时才给 trainer 注入 `--resume`；
- completed trial：检测到 `result.json` + adapter artifact 后直接跳过训练；
- matrix plan：保存 `plan_fingerprint_sha256`，改变模型、变体、seed 或超参后不能混入原矩阵续跑。

真实 MetaDrive scene_train-only launcher smoke 已验证 fresh-with-resume 与 completed-skip 两条路径。该机制用于提高长实验的可恢复性，不改变 validation13/release16 的隔离规则。
