# v3.3.0-dev 开发验收矩阵

> 本节验收的是**开发链路和实验治理**，不是模型 promotion。v3.3 当前没有 release 性能声明。

| 需求 | v3.3 当前实现/证据 | 判定 |
|---|---|---|
| MetaDrive 0.4.3 runtime | 用户提供 matching assets；headless reset + 64 steps；Python 3.13.5 / Panda3D 1.10.16 / Gymnasium 1.2.3 / pygame 2.6.1 | **通过** |
| 项目 state-policy 闭环 | frozen v2.8 + identity v3.3，在真实 MetaDrive 两 episode 运行；62-D state、finite action/diagnostics | **通过** |
| Temporal episode boundary | history 1→6；第二 episode reset 后从 0/1 重新开始；evaluator warm-up 后强制 reset | **通过** |
| Neighbor representation | masked set-attention + permutation-invariance tests；C/D 真实短训练有非零 attention | **工程通过；性能待验证** |
| Mini DAgger | 200 real-MetaDrive transitions；protected supervisor labels 15；train/save/reload 全链路成功 | **通过** |
| Raw deployment | RouteAssist OFF、SafetySupervisor OFF；runtime interventions=0；applied action=agent action | **通过** |
| Representation ablation | A/B/C/D 仅改变 history{1,6} × neighbors{off,on}，四组 risk-loss=0 | **协议通过** |
| Risk auxiliary ablation | R0/R15 固定 D representation 单独比较；主训练默认 risk-loss=0 | **协议通过** |
| Reproducibility | 显式 `--training-seed` + strict `run_config.json` fingerprint；错误 resume 被拒绝 | **通过** |
| Selection leakage | validation12 seed 38000 已工程消费并永久降级；selection 改为 fresh validation13 | **治理修复完成** |
| Fresh selection | validation13=40000–40199 尚未打开 | **待正式训练矩阵完成** |
| Final holdout | release16=39000–39199 尚未打开，launcher 不提供 release16 selection 入口 | **保持封存** |
| Scene-train pilot | A/B/C/D + R15 二轮 learner-distribution pilot 完成；仅作训练稳定性诊断 | **通过；不作 promotion 证据** |
| 全量回归 | 323 passed / 1 skipped；含 9 个真实 MetaDrive marker tests | **通过** |
| 安全/实车 | 无形式化安全、SIL/HIL、法规认证或真实道路验证 | **不通过/不声称** |

## 当前晋升状态

**NOT PROMOTED.** 下一 gate 是完成预注册多-seed training matrix 后首次打开 validation13；在 selection + freeze 完成前不得打开 release16。

---

# v3.2.0 验收矩阵

| 需求 | v3.2 实现 | 证据 | 判定 |
|---|---|---|---|
| Curve/ramp learned lateral | CurveRouteAdapter direct residual + route-intent encoder | `models/metadrive_curve_route_adapter_v32.pt` | **真实训练通过** |
| 训练数据 | curve_train 34000–34011，4,417 MetaDrive transitions | `experiments/v3_2/curve_route_adapter_v2/result.json` | **通过** |
| Runtime无几何规则转向 | primary release15 RouteAssist=off | `reports/v3_2/comparison.json` | **通过** |
| Runtime无SafetySupervisor | primary release15 safety=off | 同上 | **通过** |
| Validation-only selection | validation11 35000–35014，候选100% success/0 catastrophic | `selection_validation11.json` | **gate通过** |
| Pre-release freeze | release15首开前冻结模型/adapter/selection/core source SHA | `release15_freeze.json` | **通过** |
| Fresh release15 | 36000–36014；93.3% success / 0 collision / 6.7% off-road | `comparison.json` | **有限fresh holdout改善** |
| Curve能力 | release15 2/3 success | stage report | **改善但未完全通过** |
| 统计强度 | 15 paired scenarios；5改善/0退步；McNemar p=0.0625 | `comparison.json` | **方向性证据，非统计显著** |
| 安全认证 | 无真实道路/法规SIL-HIL/多训练seed安全验证 | known limitations | **不通过/不声称** |
| 工程回归 | 298 non-MetaDrive + 8 real MetaDrive tests | `reports/v3_2/release_tests.json` | **306 passed / 1 skipped / 0 failed** |
| 外部CWD复现 | consumed validation11 curve smoke；无RouteAssist/SafetySupervisor | `reports/v3_2/cli_smoke.json` | **通过** |
| Python分发 | 3.2.0 wheel + fresh-target import | release verification | **通过** |

## v3.2 解释边界

- 93.3% success 是 frozen base PPO + learned CurveRouteAdapter 的有限 MetaDrive holdout 结果，不是实车安全率。
- Teacher 是训练期 hand-designed pure-pursuit；runtime rule 已移除，但能力仍来自 imitation/DAgger。
- curve 仍有 1/3 off-road；release15 已消费，后续不得用于调参。

---

# v3.1.0 需求—实现—验收矩阵

版本号表示研究工程发行阶段，不表示自动驾驶安全认证。v3.1首次把v3.0运行时pure-pursuit RouteAssist的部分横向修正蒸馏为**learned residual adapter**；raw adapter评估不调用几何RouteAssist。

| 需求 | 当前实现 | v3.1 证据 | 判定 |
|---|---|---|---|
| FR1 MetaDrive真实仿真 | MetaDrive 0.4.3真实reset/step/训练/评估 | v2.3+ integration + v3.1 DAgger/release14 | **工程通过** |
| FR2 learned lateral transfer | state+base action → learned residual steering + learned blend | 3轮DAgger，3,634 transitions/15 train seeds | **训练链路与fresh holdout改善通过；curve仍失败** |
| FR2 runtime rule removal | raw adapter推理不调用pure-pursuit | adapter evaluator/env route_assist=false + integration test | **通过** |
| FR2 longitudinal safety | 可选SafetySupervisor，仅修改纵向 | release14 supervised | **系统组件通过；不是raw learned性能** |
| FR3 validation10 selection | 32000–32014，base/adaptor/supervised/reference | raw success 33.3→46.7%，catastrophic 66.7→53.3% | **promotion gate通过；现已消费** |
| FR3 pre-release freeze | release14前冻结base/adapter/config/selection/core source SHA | `reports/v3_1/release14_freeze.json` | **通过** |
| FR3 release14 fresh holdout | 33000–33014，15个capability scenarios | raw adapter 60% success / 6.7% collision / 33.3% off-road | **有限learned-system证据** |
| FR3 paired统计 | 相同stage/seed配对 | 4改善/0退步/5双好/6双坏，McNemar p=0.125 | **方向改善但未达统计显著** |
| FR3 capability分解 | curve/ramp/bottleneck/intersection/dense | ramp 3/3；dense 2/3；curve 0/3 | **能力不均衡，curve不通过** |
| FR3 learned adapter审计 | learned blend/correction/active fraction进入逐回合与聚合指标 | release14 mean blend 14.8%，active 98.0% | **通过；这是神经残差，不是规则权限** |
| 工程交付 | tests/docs/adapter registry/system registry/wheel/manifest/ZIP | final verification | 以机器验证为准 |

## v3.1 应如何解释

- 训练teacher是手写RouteAssist，但**raw release14推理没有调用pure-pursuit**。
- release14 raw base→adapter：success 33.3%→**60.0%**，collision 26.7%→**6.7%**，off-road 40.0%→**33.3%**。
- paired outcome只有4改善/0退步且n=15，exact McNemar p=0.125；不能宣称统计显著。
- curve tracking仍0% success /100% off-road，说明横向能力没有整体解决。
- Adapter+SafetySupervisor 66.7% success/0 collision属于learned lateral + rule longitudinal系统。
- release14已经消费；下一轮必须使用新的validation11/release15或更晚域。

## 不应扩大解释

- 60% success ≠ 60%真实道路安全率。
- learned residual ≠ 自主发现的规划算法；它来自规则teacher蒸馏。
- 98% adapter active ≠ 98%规则介入；adapter本身是learned module。
- 当前项目仍禁止用于实车控制。

---

# v3.0.0 需求—实现—验收矩阵

版本号表示研究工程发行阶段，不表示自动驾驶安全认证。v3.0 没有晋升新 learned checkpoint；它在冻结 v2.8 actor 上增加透明 **adaptive shared-control RouteAssist**，目标是减少 v2.9 固定 pure-pursuit 的转向规则权限，并用新的 validation9/release13 验证。

| 需求 | 当前实现 | v3.0 证据 | 判定 |
|---|---|---|---|
| FR1 MetaDrive真实仿真 | MetaDrive 0.4.3真实 reset/step、训练、评估 | v2.3+ integration + v3.0 release13 | **工程通过** |
| FR2 PPO/SAC/MAPPO | PPO/SAC/MAPPO链路与历史模型保持可运行 | registry/tests | **工程链路通过；raw安全未通过** |
| FR2 raw learned lateral | 冻结v2.8 actor；无v3 learned checkpoint | release13 raw: 26.7% success / 66.7% off-road | **能力不通过** |
| FR2 adaptive RouteAssist | policy-visible curvature/heading/lane need score；blend∈[0,0.5] | validation9/release13 + tests | **透明系统组件通过；非learned policy** |
| FR2 longitudinal safety | SafetySupervisor只改纵向动作 | release13 metrics | **系统组件通过；非安全认证** |
| FR3 validation9 | 29000–29014 fixed vs adaptive | adaptive 93.3% success / 13.7% mean blend | **用于选择，现已消费** |
| FR3 pre-release freeze | release13前冻结model/config/core source/selection SHA | `reports/v3_0/release13_freeze.json` | **通过；hash/chronology均通过** |
| FR3 release13 fresh holdout | 30000–30014 raw/fixed/adaptive三系统 | adaptive 80% success / 0 collision / 20% off-road | **有限系统级证据** |
| FR3 shared-control审计 | 逐步记录need/blend/proposed/applied action；聚合learned steering authority | adaptive mean blend 11.4%，authority 88.6% | **通过；仍是hybrid** |
| FR3 paired统计 | 相同stage/seed配对 | fixed→adaptive 1改善/1退步，McNemar p=1.0 | **未证明success优于fixed** |
| FR3 capability分解 | curve/ramp/bottleneck/intersection/dense | curve adaptive 1/3 vs fixed 2/3 | **存在stage trade-off** |
| 工程交付 | tests/docs/registries/manifest/wheel/ZIP复验 | 最终verification | 以机器验证为准 |

## v3.0 应如何解释

- v3.0 **不是新RL模型**；`models/registry.json` 没有新增v3 learned checkpoint。
- 相比v2.9固定RouteAssist，release13 aggregate success/catastrophic相同，但平均几何steering blend从100%降到**11.4%**，平均learned steering authority为**88.6%**。
- RouteAssist仍在约**76.5%**时间步产生非零修正，且SafetySupervisor约**14.1%**时间步修改纵向动作，所以它仍是neural+rule shared-control system。
- adaptive route completion低于fixed（84.5% vs 92.9%），curve stage也更差；不能只报告authority改善而忽略能力trade-off。
- validation9与release13均已消费，后续不得用于新的参数选择。

## 不应扩大解释

- shared-control 80% success ≠ raw PPO 80% success。
- 88.6% mean steering authority ≠ neural actor已学会pure-pursuit能力。
- 15个release13场景 ≠ 自动驾驶安全认证或真实事故率估计。
- 当前项目仍禁止用于实车控制。

---

## 历史 v2.9 验收矩阵


版本号表示研究工程发行阶段，不表示自动驾驶安全认证。v2.9 在真实 MetaDrive 0.4.3 中新增 route schema v4、透明 lateral RouteAssist、hybrid rule+neural evaluation，以及 **真正写在 release12 之前的 freeze manifest**。没有为了版本升级而晋升表现不稳定的 learned lateral checkpoint。

| 需求 | 当前实现 | v2.9 证据 | 判定 |
|---|---|---|---|
| FR1 MetaDrive真实仿真 | 0.4.3源码/assets/wheelhouse；真实 reset/step、PPO/MAPPO/BEV/RGB/ScenarioEnv | v2.3+ evidence + integration tests | **工程通过** |
| FR1 Gymnasium接口 | native/MetaDrive五返回语义与真实checker路径 | tests | **通过** |
| FR2 PPO | MetaDrive state PPO 32,768 world steps | `models/metadrive_ppo_state_32768.pt` | 训练通过；raw安全能力不通过 |
| FR2 SAC | native SAC完整replay/optimizer/RNG恢复 | v2.1 tests | native通过；MetaDrive长训练未正式验收 |
| FR2 MARL | MetaDrive两学习车同一物理世界 CTDE MAPPO 8,192 steps | v2.3 evidence | 链路通过；复杂协作不足 |
| FR2 route schema v4 | 按route arc-length排序/采样的前向route preview；schema3兼容 | `test_v29_route_observation.py` | **接口与回归通过** |
| FR2 learned lateral adaptation | schema4 expert/weighted DAgger/ridge head | `experiments/v2_9/*` | **多条真实实验完成；闭环负结果，不晋升** |
| FR2 transparent lateral controller | policy-visible pure-pursuit RouteAssist，只改steering | `src/navigation/route_assist.py` | **系统组件通过；不是learned policy** |
| FR2 longitudinal safety | SafetySupervisor只改longitudinal command | v2.4+ + v2.9 integration | **系统组件通过；不是安全证明** |
| FR2 semantic BEV | BEV+motion+history CNN | MetaDrive 4,096-step PPO | 反传链路通过；驾驶能力失败 |
| FR2 raw RGB | MetaDrive RGBCamera + motion + CNN PPO/BC | v2.4 evidence | 训练链路通过；驾驶能力失败 |
| FR3 capability suite | curve/merge/bottleneck/intersection/dense progress | `src/evaluation/capability_suite.py` | **真实运行** |
| FR3 seed隔离 | route_train / validation8 / release12 与历史域互斥 | generator + split tests | **通过** |
| FR3 validation8 | 26000-26014，raw / route / hybrid对比 | `reports/v2_9/validation8_*` | hybrid选定；新learned checkpoint未promote |
| FR3 pre-release freeze | release12前冻结模型、配置和核心源码SHA | `reports/v2_9/release12_freeze.json` | **通过；修复v2.8流程瑕疵** |
| FR3 release12 fresh holdout | 27000-27014，raw vs hybrid | raw 33.3/13.3/53.3 → hybrid 93.3/0/6.7 | **有限仿真系统级改善** |
| FR3 paired统计 | 相同stage/seed paired success | 10改善/1退步/4双好，McNemar p=0.0117 | 描述性研究证据；非安全认证 |
| FR3 intervention审计 | route/safety介入时间与修正幅度分开统计 | route 99.82%，safety 15.40% on release12 | **透明报告；禁止归因为raw RL** |
| FR3 失败诊断 | collision/off-road/timeout、route completion、lane/jerk/latency等 | reports | **研究评估通过** |
| FR4 程序化场景 | MetaDrive map/density/seed + capability curricula | configs/suites | **通过** |
| FR4 真实数据 | assets内3个Waymo/ScenarioNet示例闭环 | historical evidence | 小样本接口通过；泛化未验收 |
| 端到端 Agent | state/BEV/RGB+motion → neural policy → control → MetaDrive | models/experiments | 研究链路通过；可靠视觉驾驶未完成 |
| 可复现性 | config/seed/model SHA/freeze/report/manifest | v2.9 artifacts | **同运行时可追踪** |
| 工程交付 | tests/docs/model+system registry/manifest/wheel/ZIP复验 | final verification | 以机器验证为准 |

## v2.9 应如何解释

- v2.9 **没有新的 curated learned model**；`models/registry.json` 仍以 v2.8 capability student 为最新 learned checkpoint。
- v2.9 新增 `models/system_registry.json`，单独登记 neural+rule hybrid system，避免把系统结果写进 learned-model registry。
- release12 fresh 15场景：raw success 33.3% → hybrid 93.3%，collision-or-off-road 66.7% → 6.7%。
- RouteAssist 在 99.82% 时间步修改 steering，说明这是**规则主导的横向系统结果**。SafetySupervisor 也在15.40%时间步参与纵向控制。
- release12 已消费，未来任何调参不得继续将27000-27199当作未见测试域。

## 不应扩大解释

- hybrid 93.3% success ≠ raw PPO 93.3% success。
- pure-pursuit RouteAssist ≠ learned path-planning能力。
- 15个release12场景和p=0.0117 ≠ 自动驾驶安全认证、真实事故率或跨训练seed统计结论。
- 3个Waymo示例 ≠ 大规模真实数据泛化。
- RGB/BEV参数更新 ≠ 视觉策略收敛。
- 项目无法规SIL/HIL、硬实时、ASIL/SOTIF认证；**禁止用于实车控制**。
