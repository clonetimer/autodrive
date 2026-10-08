# v3.3.0-dev 已知限制与实验边界

| 范围 | 当前真实状态 |
|---|---|
| 版本状态 | **development candidate，不是已晋升 release**；没有注册新的 v3.3 推荐模型，也没有 v3.3 release 性能声明。 |
| Representation 结论 | Scene-aware temporal adapter 已在真实 MetaDrive 完成 A/B/C/D 短训练与 scene-train 二轮 pilot；D 在 iteration-1 diagnostic teacher-fit MSE 最低，但这是训练域诊断，不是泛化或成功率证据。 |
| Risk auxiliary | R15 pilot 在相同 D representation 下把 iteration-1 steering MSE 从约 3.87e-3 恶化到 5.75e-3，且未减少 1/4 off-road；risk label 高度不平衡。因此默认 risk-loss=0，R15 仅保留为显式 opt-in ablation。 |
| Selection holdout | `validation12` 的 seed 38000 已用于 evaluator 工程 smoke，并又做过 v3.2 sanity baseline，**永久视为 engineering-consumed**；正式 candidate selection 已迁移到未打开的 `validation13=40000–40199`。 |
| Final holdout | `release16=39000–39199` 尚未打开；在 candidate 通过 validation13 并冻结 artifact/config/core-source hash 之前禁止访问。 |
| Source actor | v3.3 仍绑定 frozen v2.8 base actor；尚未完成 3+ 独立 source-PPO lineage，当前 3-seed 计划主要覆盖 adapter initialization，而非完整算法跨 source-policy seed。 |
| DAgger distribution | protected/raw deployment 链路已验证，但正式 mixed-support DAgger 尚未完成；短 pilot 仍显示 iteration-1 learner distribution 上 supervised fit 明显变难。 |
| Neighbor supervision | pure-pursuit steering teacher 不显式依赖邻车；neighbor branch 是否提升闭环 interaction 必须靠 clean A/B/C/D + 后续 RL/interaction objective 验证，不能由 attention mass 本身推断。 |
| 统计强度 | 当前 scene-train pilot 只有 1 个 adapter training seed、短 horizon；不允许用其差异做统计显著性或模型晋升结论。 |
| Safety | 真实 MetaDrive smoke、0 hidden runtime interventions 等只证明工程路径；不构成事故率估计、形式化安全或实车依据。 |
| 生产用途 | 无法规 SIL/HIL、硬实时车辆接口、ASIL/SOTIF 认证；**禁止用于实车控制**。 |

## v3.3 下一阶段优先级

1. 完成预注册 A/B/C/D × training seeds 3300/4300/5300 的 scene-train 训练矩阵；四组统一 risk-loss=0。
2. 独立完成 R0/R15 × 3 seeds 风险辅助消融，不与 representation 结论混合。
3. 训练矩阵全部完成后，**首次**打开 fresh `validation13` 做 paired candidate selection；只依据预注册指标选择。
4. candidate/config/source SHA 冻结后才首次打开 `release16`，并一次性完成 final holdout。
5. 若 v3.3 representation 增益稳定，再进入 v3.4 mixed-support DAgger / v4.0 expert-guided online PPO；否则优先修 distribution mismatch，不扩大网络。

---

# v3.2.0 已知限制与未通过目标

| 范围 | 当前真实状态 |
|---|---|
| Fresh holdout规模 | release15仅15个 capability scenarios；paired McNemar p=0.0625，不能声称统计显著。 |
| Curve tracking | 由v3.1的0/3提升到v3.2的2/3，但仍有1个off-road，曲线能力未完全通过。 |
| Teacher来源 | 训练使用fixed 0.5 pure-pursuit teacher；primary runtime已无RouteAssist，但学习信号仍来自hand-designed controller。 |
| Base PPO lineage | adapter绑定v2.8 base actor；核心base PPO主要来自单一训练谱系，未完成3+独立source-seed复现。 |
| Safety | release15 0 collision仅为15场景有限模拟结果；不构成事故率估计、安全认证或实车依据。 |
| Visual policy | BEV/RGB真实训练链路存在，但当前高表现v3.2能力仍建立在state-based base policy之上。 |
| MARL/真实数据 | 保留真实MetaDrive MAPPO/Waymo小样本链路；复杂协作与规模化真实场景泛化仍不足。 |
| Panda3D生命周期 | fresh-process/resume降低偶发teardown卡顿，但无法从根本上消除引擎生命周期问题。 |
| 生产用途 | 无法规SIL/HIL、硬实时、ASIL/SOTIF认证；**禁止用于实车控制**。 |

## 下一阶段优先级

1. 新建validation12/release16，release15永久只作历史证据。
2. 训练至少3个独立source PPO seeds，并分别训练curve adapter，报告跨seed分布。
3. 针对剩余curve off-road研究temporal/recurrent route encoder与困难样本主动采集。
4. 扩大curve topology、merge冲突和ScenarioNet/Waymo场景覆盖。
5. 将同样的route-intent/DAgger能力迁移到BEV/RGB视觉策略。

---

# v3.1.0 已知限制与未通过目标

| 范围 | 当前真实状态 |
|---|---|
| Learned adapter总体能力 | fresh release14 raw success **60.0%**、collision **6.7%**、off-road **33.3%**；比base明显改善，但仍有40% catastrophic failure。 |
| Curve tracking | release14仍 **0/3 success / 3/3 off-road**；这是当前最明确的横向短板。 |
| 统计强度 | 15个fresh scenarios；base→adapter paired success为4改善/0退步，exact McNemar **p=0.125**，不足以宣称统计显著。 |
| Teacher依赖 | adapter训练来自adaptive pure-pursuit teacher；runtime rule已移除，但能力来源仍是imitation，不是自主规划发现。 |
| Source actor | base仍是v2.8 capability actor，核心source PPO主要来自单训练seed；未完成3+独立source-seed复现。 |
| Longitudinal安全 | adapter+SafetySupervisor为66.7% success/0 collision，但SafetySupervisor仍是rule layer；不能计作raw learned safety。 |
| Adaptive reference差距 | v3.0 adaptive rule reference在相同release14为80% success/20% off-road；learned adapter尚未完全复制teacher闭环能力。 |
| RGB/BEV | 真实训练链路存在，但可靠视觉闭环仍未完成。 |
| MARL/真实数据/SAC | 保留历史工程链路；复杂多智能体协作、大规模ScenarioNet/Waymo、MetaDrive SAC长训练仍不足。 |
| 生产用途 | 无法规SIL/HIL、故障注入、硬实时、ASIL/SOTIF认证；**禁止用于实车控制**。 |

## 下一阶段优先级

1. 新建`validation11/release15`，release14只保留历史证据。
2. 建立curve-specialist learned adapter / recurrent route encoder，目标是**curve off-road显著下降且不回退ramp/dense**。
3. 用至少3个独立source PPO training seeds重复adapter蒸馏，报告跨seed分布。
4. 将adapter uncertainty作为显式估计量，研究learned arbitration/low-frequency fallback，并单独报告fallback比例。
5. 继续扩大真实MetaDrive/ScenarioNet以及BEV/RGB训练，避免项目停留在state-policy teacher distillation。

---

# v3.0.0 已知限制与未通过目标

| 范围 | 当前真实状态 |
|---|---|
| Raw learned policy安全 | 冻结v2.8 actor在release13仅 **26.7% success / 6.7% collision / 66.7% off-road**；raw lateral能力明显未解决。 |
| Adaptive系统归属 | v3.0 adaptive system为 **80% success / 0 collision / 20% off-road**，但RouteAssist约 **76.5%** 时间步有非零steering修正，SafetySupervisor约 **14.1%** 时间步修改纵向动作；不是raw RL性能。 |
| 规则权限 | 平均几何steering blend降到 **11.4%**、mean learned authority **88.6%**，但这是控制权占比，不是学习能力证明。 |
| Fixed对比 | adaptive与fixed aggregate success/catastrophic相同；route completion更低（84.5% vs 92.9%），curve stage更差（1/3 vs 2/3 success）。 |
| Holdout规模 | release13只有15个fresh capability scenarios；fixed→adaptive paired McNemar p=1.0。 |
| release数据 | validation9/release13均已查看并消费；下一轮必须新建validation10/release14或更晚域。 |
| 新learned模型 | v3.0没有新checkpoint；`models/registry.json`保持v2.8为最新相关learned capability model。 |
| RGB/BEV | 训练链路真实，但可靠视觉闭环仍未完成。 |
| MARL/真实数据/SAC | 保持历史工程链路；复杂MARL、规模化Waymo/ScenarioNet、MetaDrive SAC长训练仍不充分。 |
| 生产用途 | 无法规SIL/HIL、故障注入、硬实时、ASIL/SOTIF认证；**禁止用于实车控制**。 |

## 下一阶段优先级

1. 在全新的 `validation10/release14` 上研究**降低非零RouteAssist intervention frequency**，不仅降低平均blend。
2. 针对curve stage单独训练lateral learner，目标是在不增加规则权限的情况下恢复fixed系统的curve完成率。
3. 训练至少3个独立source PPO seeds，报告跨seed均值/置信区间；当前核心actor仍主要来自单训练seed。
4. 尝试将adaptive teacher correction用于online DAgger/PPO fine-tune，并把“teacher intervention fraction下降”设为显式训练/晋升指标。
5. 扩展更大MetaDrive/ScenarioNet场景和raw RGB/BEV训练，再讨论端到端视觉能力。

---

## 历史 v2.9 限制


| 范围 | 当前真实状态 |
|---|---|
| Raw learned policy安全 | v2.9没有晋升新 learned checkpoint。冻结v2.8 raw policy在 release12 15场景只有 **33.3% success / 13.3% collision / 53.3% off-road**，明显不安全。 |
| Hybrid结果归属 | v2.9 hybrid 为 **93.3% success / 0% collision / 6.7% off-road**，但 RouteAssist 在 **99.82%** 时间步介入 steering，SafetySupervisor 在 **15.40%** 时间步介入 longitudinal control。绝不能写成raw RL性能。 |
| Learned curve/ramp能力 | schema-v4 pure-expert、steering-weighted、DAgger与lateral-head ridge均实际训练/拟合，但validation8 raw闭环仍高出界；没有新learned模型通过晋升。 |
| Route schema v4 | 解决了hairpin/回折路线预览排序问题，但仅是观测表征改进；没有证据证明它单独提升闭环性能。 |
| Pure-pursuit边界 | RouteAssist只利用policy-visible route preview，但它仍是手写几何控制器；接近100%的介入说明横向neural policy没有承担主要控制职责。 |
| Holdout规模 | release12只有15个fresh MetaDrive capability场景。paired success 10改善/1退步，McNemar p=0.0117，只能视为有限模拟证据。 |
| 跨训练seed | 核心MetaDrive source PPO仍主要来自单训练seed；完整算法提升未用3+独立source seeds重复。 |
| release数据 | release12已查看并永久消费；不能继续用于模型选择或调参。下一轮必须新建validation9/release13。 |
| Dense/interaction | hybrid在本轮3个dense holdout均成功，但样本极少，不构成复杂交通可靠性证明。 |
| SafetySupervisor | 是独立规则层，不是CBF、形式化验证或法规级安全功能。 |
| RGB PPO | 真实RGB PPO只做短训练；CNN参数更新但不具备可靠闭环驾驶能力。 |
| RGB输入 | `rgb_ego` = RGB + ego/navigation motion，不是camera-only端到端。 |
| BEV | semantic BEV来自仿真真值，短训练策略严重出界。 |
| MARL | 两车MAPPO真实训练已跑通，但复杂merge/yield/intersection协作任务不足。 |
| 真实数据 | 仅3个assets自带Waymo转换场景小回归；不构成ScenarioNet/Waymo规模泛化。 |
| SAC | 完整学习状态恢复在native验证；MetaDrive SAC长训练未正式完成。 |
| 长训练 | segmented resume恢复模型/optimizer/counter，不恢复Panda3D逐帧世界快照。 |
| Panda3D生命周期 | fresh-process与commit marker降低长进程风险，但不能消除teardown偶发滞留。 |
| CARLA | 未集成；当前高保真后端为MetaDrive。 |
| 生产用途 | 无法规SIL/HIL、故障注入、硬实时车辆接口、安全冗余或认证。**禁止用于实车控制。** |

## 下一阶段优先级

1. 新建 `validation9/release13`，release12只作为历史证据。
2. 不再把 pure-pursuit 作为“算法提升”，而是以它作为 lateral teacher / safety baseline，目标是**显著降低 RouteAssist intervention fraction**。
3. 训练至少3个独立source PPO seeds，统一比较 raw policy、teacher-distilled policy 与 hybrid system。
4. 对 curve/ramp 做 online DAgger / PPO fine-tune，专门采样 student 偏离路线后的状态，解决离线MSE低但闭环分布偏移大的问题。
5. 将路径预览扩展为曲率、lane connectivity、merge conflict 等拓扑表示；做 ablation 证明观测改进的独立贡献。
6. 扩大ScenarioNet/Waymo和RGB/BEV训练数据后，再推进视觉闭环能力。
