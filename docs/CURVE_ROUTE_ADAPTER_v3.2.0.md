# v3.2.0 Curve-Aware Learned Route Adapter

## 目标

v3.1 已将部分 RouteAssist 能力蒸馏成神经残差，但 fresh release14 的 curve tracking 仍为 0/3 success。本版本不增加运行时规则接管，而是针对曲线和匝道分布偏移，改造 learned adapter 的表示与训练目标。

## 方法

`CurveRouteAdapter` 保持冻结的 v2.8 PPO actor，输入仅包含 policy-visible ego/route state 与 base action。相比 v3.1 的 residual+blend 分解，v3.2 使用直接 steering residual head，并加入 route-intent encoder；route preview 的角度/角差只作为神经输入特征，不在 release runtime 执行 pure-pursuit。

训练使用真实 MetaDrive `curve_train` 34000–34011，fixed-blend pure-pursuit 仅作为 DAgger teacher，SafetySupervisor 仅用于采集期间纵向防碰撞。三轮 teacher mixing 1.0/0.5/0.2，共采集 **4,417 transitions / 12 unique seeds**。最终 adapter SHA-256：`4230d110459468a31cad87bb3fa12fe556f3a951e66ed880e5d5ce3c44bb92a4`。

## 防泄漏流程

- candidate selection：`validation11` 35000–35014；
- selection gate 在 release15 未打开时完成；
- `reports/v3_2/release15_freeze.json` 在 release15 第一次请求之前冻结 base model、adapter、selection、核心源码 SHA；
- pre-release route-adapter registry 内容另存为 `reports/v3_2/frozen_route_adapter_registry.json`，其 SHA 与 freeze manifest 中记录一致；
- freeze 后才首次打开 `release15` 36000–36014；release15 已永久消费。

## Validation11

相同 15 个 capability 场景：

| Policy | Success | Collision | Off-road | Catastrophic |
|---|---:|---:|---:|---:|
| v3.1 raw adapter | 60.0% | 6.7% | 33.3% | 40.0% |
| **v3.2 raw adapter** | **100%** | **0%** | **0%** | **0%** |

候选通过预声明 gate 后冻结。Validation11 只用于选择，不作为最终泛化结论。

## Fresh Release15

Primary runtime **不开启 RouteAssist，也不开启 SafetySupervisor**。

| Metric | v3.1 raw | **v3.2 raw** | Delta |
|---|---:|---:|---:|
| Success | 60.0% | **93.3%** | +33.3pp |
| Collision | 13.3% | **0%** | -13.3pp |
| Off-road | 26.7% | **6.7%** | -20.0pp |
| Collision OR off-road | 40.0% | **6.7%** | -33.3pp |
| Route completion | 73.45% | **92.41%** | +18.96pp |
| Lane abs error | 0.435 m | **0.196 m** | -0.240 m |

相同场景 paired success：**5 failure→success / 0 success→failure / 9 both success / 1 both failure**。Two-sided exact McNemar `p=0.0625`。方向明显，但 15 场景样本仍不足以宣称统计显著或安全认证。

### 分能力结果

| Stage | v3.1 success | **v3.2 success** | v3.2主要失败 |
|---|---:|---:|---|
| curve tracking | 0/3 | **2/3** | 1 off-road |
| ramp merge | 2/3 | **3/3** | none |
| bottleneck yield | 2/3 | **3/3** | none |
| intersection yield | 3/3 | **3/3** | none |
| dense progress | 2/3 | **3/3** | none |

Curve tracking 已从 0/3 提升到 2/3，但仍未解决；不能将 93.3% aggregate success 解读为曲线能力完全通过。

## 解释边界

1. 训练 teacher 是手写 pure-pursuit；这属于 imitation / DAgger transfer，不是策略自主发现几何规划规律。
2. release15 primary result 的运行时没有几何 RouteAssist 或 SafetySupervisor，因此可归因于 frozen base actor + learned adapter 组合；但 base actor 仍主要来自单一 training lineage。
3. 15 个 fresh MetaDrive 场景不是安全认证、真实事故率估计，也不足以支持跨训练 seed 的统计结论。
4. 当前仍禁止用于实车控制。

## 下一步

- 新建 validation12/release16，release15 只保留历史证据；
- 训练至少 3 个独立 source PPO seeds，并分别训练/评估 curve adapter；
- 针对剩余 curve off-road 引入 recurrent route encoder / temporal preview，但保持 runtime 无规则接管；
- 扩大 curve/ramp topology 与 ScenarioNet/Waymo 场景覆盖；
- 继续推进 BEV/RGB learned policy，使 state teacher 不成为唯一能力来源。
