
### 2026-09-09 — v3.3 长矩阵恢复机制补强

- 正式默认规模 trial 在真实 MetaDrive 中可超过单次短执行窗口；一次 A/3300 formal-default 尝试在 iteration-1 中途被外层超时终止，但 iteration-0 与已完成 worker 均正确 commit，可由 trainer resume 复用。
- 修复 matrix-level `--resume`：不再向从未开始的 trial 无差别传递 resume。fresh/partial/completed job 分别新建/续跑/跳过。
- 新增 `plan_fingerprint_sha256`，防止改变 ablation 参数后把旧 trial 混入同一矩阵。
- 真实 launcher smoke：全新目录带 `--resume` 能完成 scene_train mini trial；相同命令第二次正确跳过 completed job；未访问 validation13/release16。
- 修复后全量回归：**323 passed / 1 skipped**。

# v3.3.0-dev 开发台账 — Scene-Aware Temporal Adapter

- 研究问题：把 v3.2 的单帧 route follower 推进为有时序/交互表示的 learned policy，同时保持 raw runtime 无规则控制接管。
- 代码：新增 temporal route GRU、neighbor set attention、diagnostic risk head、v3 artifact 保存/加载与 state-history wrapper。
- 训练：新增 v3.3 DAgger script 和 episode-safe history builder；dataset schema 已为后续 expert-guided PPO / constrained RL 扩展。
- 评估：修复 warm-up history contamination；新增 risk/attention/visible-neighbor/history diagnostics 聚合。
- 数据隔离：新增 scene_train 37000–37999 / validation12 38000–38199 / release16 39000–39199；当前没有 v3.3 release16 结果。
- 回归：v3.3 定向 6 tests passed；全量 non-MetaDrive 301 passed / 4 skipped / 8 deselected。
- 额外诊断：在旧 v3.2 DAgger 数据上做过仅用于代码路径验证的离线 fit smoke；它不构成 v3.3 candidate selection，也不允许用于 release16 结论。
- 阻塞：本轮执行容器没有工程内 MetaDrive assets/runtime，真实 MetaDrive v3.3 rollout 尚未复验；不得标记为训练完成或安全晋升。

# v3.2.0 开发台账 — Curve-Aware Learned Route Adapter

- 目标：解决 v3.1 release14 curve 0/3 的横向失败，同时保持 primary runtime 无几何 RouteAssist。
- 实施：新增 direct-residual curve adapter、route-intent encoder、curve/ramp focused DAgger、resume/fresh-worker 执行、validation-only gate。
- 训练：真实 MetaDrive 4,417 transitions / 12 curve_train seeds；teacher 仅训练期执行。
- Validation11：15/15 success，0 collision/off-road；v3.1 同域 9/15 success。
- Freeze：release15 首开前冻结 base/adapter/selection/core source SHA；预发布 registry 内容另存不可变快照。
- Release15：14/15 success、0 collision、1 off-road；v3.1 为9/15 success、2 collision、4 off-road；paired 5改善/0退步，p=0.0625。
- 失败保留：curve stage 仍 2/3，seed 36001 off-road；不标记为安全完成。
- 收尾：306 passed / 1 skipped / 0 failed（298 non-MetaDrive + 8 real MetaDrive）；non-MetaDrive src line coverage 85.79%；外部CWD MetaDrive/通用CLI均通过；3.2.0 wheel构建并fresh-target导入通过；最终ZIP独立复验见旁车verification JSON。

# 开发台账 — 本轮真实执行

本文件仅记录本轮有代码/测试/产物证据的开发，不继承此前对话中的“已完成”口头结论。时间以产物内 UTC/runtime 记录为准。

| 阶段 | 实质变更 | 验证证据 | 完成判定 |
|---|---|---|---|
| 输入审计 | 检查旧双根ZIP、MetaDrive源码、依赖及已有权重 | INPUT_AUDIT.json；源码/API静态审阅 | 完成审计；旧版本完成状态不可信赖 |
| MetaDrive真实接入尝试 | 提取0.4.3源码，实际尝试pip安装；实现真实单/多车/ScenarioEnv适配 | dependency_install_failed.log；metadrive_doctor.json；metadrive_attempt/failure.json | **运行阻塞，不是完成真实训练**；零MetaDrive步 |
| CPU物理与算法重构 | 共享世界、IDM、自行车模型、碰撞检测；真正PPO/IPPO/MAPPO/SAC、CNN与混合actor | tests_initial.xml：69 passed /5 skipped；五种算法/输入短训练 | native链路通过；依赖门禁保留 |
| 外部适配与轨迹修复 | MetaDrive终止销毁前抓取快照；回放轨迹按生命周期出现/消失 | tests_post_adapters.xml：69 passed /5 skipped | native/单元通过；真实MD测试仍跳过 |
| 实际实验矩阵 | 三种子PPO、MAPPO、BEV、混合、SAC、奖励/IPPO对比共10个真实训练 | experiments/release/matrix.json；真实model.pt/CSV/TFRecord | 180224世界步；不是只创建配置 |
| 课程与配置 | YAML include循环检查；以联合回合真实结果升级的课程接入训练与resume | tests_after_curriculum.xml：73 passed /5 skipped | 集成通过；随后65536步未达晋级阈值，阶段0如实保留 |
| 文件与统计测试 | protobuf独立解码、Wilson场景区间、数据校验、延迟轨迹 | tests_full_features.xml：74 passed /5 skipped | 测试通过，无MetaDrive虚假pass |
| 评估与可视化 | 固定测试20场景、独立验证诊断、六种泛化测试、真实GIF与轨迹 | reports/release_evaluation；reports/figures；videos | 失败案例全部保留；高碰撞表现未通过行为目标 |
| 性能 | 区分step调用/含reset循环/策略推理，不混淆渲染FPS | reports/performance.json；logs/performance.log | 3×500步/输入模式；此前2000步长调用超时不用于结果 |
| 发布收尾 | 全量带覆盖率测试、CLI实际执行、独立解压复验、SHA256文件清单、单根ZIP | 最终reports/RELEASE_STATUS.json及packaging_evidence.json | 由机器门禁决定；不得把research preview说成全部目标完成 |

## 重要修复与工程决策

1. 不提供伪造 Gym 包；真实 Gymnasium 存在时使用其 Env/Space，不存在时明确用离线协议，并跳过真实checker测试。
2. PPO使用on-policy rollout，SAC单独实现replay，纠正原设计统一Replay Buffer的混淆。
3. 截断bootstrap用终止前next观测，GAE不跨回合；MARL逐Agent mask以及central critic输入有专门测试。
4. BEV CNN参数确实参与优化；actor不读取global state，禁止借“集中训练”把真值全局输入泄漏到执行端。
5. 版本号与验收状态分开：文件可叫v2.0.0，验收状态仍为CPU research preview；MetaDrive/原始RGB端到端/可靠密集交通未达标。

## 可持续执行入口

`run_experiments.py`、`evaluate_release.py`、`performance.py`、`render_reports.py` 都是实脚本，执行会产生真实产物。后台进程只在本次工具执行过程中使用，并在打包前确认退出；不会在最终回复后继续开发或训练。


# v2.1.0 实际改进台账（2026-09-07 UTC）

本轮从交付v2.0完整ZIP恢复源代码，先实际跑基线79通过/5跳过，再修改。所有数字来自logs/v2_1与reports/v2_1，不继承未经验证的口头进度。

| 内部阶段 | 实质工作 | 全量验证/产物 | 结束判定 |
|---|---|---|---|
| 基线与环境 | 恢复原ZIP、验证依赖、检查先前审计 | baseline_tests.log；network_probe.json | 基线有效，真实MD仍阻塞 |
| 奖励/运动观测 | version3奖励、风险与dt；20维motion和最多8帧ego配准历史；真实encoder接线 | reward_vision_regression.log；test_v21_reward/observation | 代码单元与优化路径通过 |
| SAC持久化 | 完整学习状态/uint8 ring replay/RNG/原子保存/配置门禁 | sac_regression.log；下一更新bitwise测试 | 学习恢复通过，非物理恢复 |
| 指标/安装 | 纵横向舒适性、time weighted p95、失败4秒；源/assets/wheel/网络验证；SB3真实入口 | metrics_full_tests.log；安装与SB3失败证据 | 内部通过，外部门禁保留 |
| 实际训练 | 旧/新奖励3种子、1/4帧、两车MAPPO、SAC首次+恢复 | matrix_plan/results；241664世界步 | 10启动+1恢复全部完成 |
| 开发评估 | 相同开发种子、多场景；全失败记录和配对统计 | 1100场景-策略/1120agent回合 | 巡航/密集改善；紧急制动/4帧相对表现退步，如实保留 |
| 工程收尾 | 127通过/5跳过，86%以上src覆盖；仓库外CLI；旧3种子权重重现；wheel；perf | tests_with_coverage.log；cli报告；legacy_reproduction；performance | 本轮CPU工程改进通过，不扩大原目标验收 |
| 发布 | 更新当前文档/保留历史、注册新模型、SHA清单、单根ZIP、独立解压复验 | ZIP旁独立verification报告为最终依据 | 研究包；无MetaDrive成功或实车安全承诺 |

本轮未开展新真实数据下载、原始相机视觉、按能力重构课程、行为克隆、安全过滤器或PettingZoo官方标准测试。模型、原始失败和旧奖励对照均保留；没有用验证集挑选最佳checkpoint。v2.0档案不被本轮覆盖。


# v2.2.0 实际部署与依赖闭环台账（2026-09-07 UTC）

| 内部阶段 | 实质动作 | 真实证据 | 判定 |
|---|---|---|---|
| 恢复与基线 | 从v2.1完整ZIP恢复，检查新上传文件及运行时 | logs/v2_2/baseline_tests.log：127通过/5跳过 | 基线有效 |
| 资源部署 | 校验0.4.3源码/assets，152文件与7处glTF引用；原子落盘、重复SHA验证 | assets_staging/reverification/final_check.json | assets阻塞已解除，非引擎运行 |
| 实际源码安装 | 创建外置venv，显式复用CPU基础库，真实pip editable安装0.4.3（无网络无依赖） | runtime_creation.json；source_install.log | 源码注册通过；七项运行库仍缺失 |
| 依赖/安装工具 | 上传内容分类、目标ABI导出、保留原CPU依赖的增量wheel收集、目标marker/依赖图/冲突/哈希检查 | tools收集器与test_v22_collector；dependency_request/plan/target.json | 离线测试通过；联网下载未验收 |
| 模拟器契约 | 修复ScenarioEnv未知配置键、转向角单位、crash_human及横向指标；新增官方观测SB3入口 | test_v22_metadrive_contract；metadrive_official/result.json | 单元通过；真实启动因依赖阻止，0step |
| 已提供真实数据 | 受限数值pickle读取、91帧时间/掩码/地图审计、summary交叉检查、录制轨迹图 | waymo_examples.json：3场景17791有效track-frame | 数据审计通过，不是仿真或RL训练 |
| 回归与收尾 | 第一轮167通过/5跳过，再加入作用域依赖检查共169通过/5跳过；仓库外PPO/MAPPO/BEV/SAC恢复CLI实跑 | tests_first_pass/after_closure/with_coverage；cli_smoke；v21_cli_regression | src覆盖86.59%；真实接口5项仍跳过 |
| 发布 | 新文档/历史归档、版本号、文件清单、ZIP、独立解压复验、模型哈希 | ZIP旁verification为最终依据 | 研究工程增量版，非MetaDrive训练完成 |

两次组合工具命令达到执行时限，已拆分后逐项重跑并保留实际成功报告；未把中断命令算作测试通过。没有正式新增训练模型，旧22个模型保留并在收尾校验。用户的安装说明ZIP不是wheelhouse，未把其联网脚本盲目执行。没有后台开发承诺，产物对应本轮实际工具调用。

# v2.3.0 MetaDrive 真实训练与传感器闭环台账（2026-09-07 UTC）

| 内部阶段 | 实质动作 | 真实证据 | 判定 |
|---|---|---|---|
| Wheelhouse验收 | 安全解包用户提供的CPython3.13/Linux x86_64 wheelhouse，检查版本、tag、哈希与依赖图 | `external/wheelhouse`；installation report | 完成，第三方wheel不进入发行ZIP |
| 隔离运行时 | 创建overlay runtime，安装缺失依赖、MetaDrive 0.4.3源码与匹配assets | `runtime_creation.json`、`metadrive_installation.json` | MetaDrive真实import/reset/40 step通过 |
| Gymnasium修复 | 新增official facade，兼容Gymnasium 1.2 reset options/seed；实际checker路径运行 | `test_metadrive_official.py` | 真实依赖环境通过 |
| 官方参考训练 | MetaDrive官方obs/reward + SB3 PPO 4096步，关闭训练env后冷加载评估 | `metadrive_official_ppo_smoke4096/result.json` | 4343真实step；策略10/10出界，如实保留 |
| 自研PPO MetaDrive | 真实训练、checkpoint恢复并完成至32768世界步 | `training_chain_32768.json`、final model | 真实训练通过；20测试75%成功/25%碰撞 |
| 长进程可靠性 | 发现单进程晚期偶发停滞，新增有界子进程分段训练；真实2×4096实验 | `segmented_metadrive_ppo_state_8192/result.json` | 工程缓解通过；非物理快照恢复 |
| MetaDrive MARL | 两学习车同世界MAPPO hybrid 8192世界步 | model/result/evaluation | 82.5% agent success；80% joint；20%场景任意碰撞 |
| MetaDrive BEV | 四帧bev_ego CNN PPO 4096步，验证encoder参数变化 | model/result/evaluation | 真实反传通过；10测试100%出界，能力失败 |
| ScenarioEnv真实轨迹 | 冻结32k PPO闭环控制assets内3个Waymo示例 | `waymo_examples_closed_loop` | 3/3成功，仅接口小样本验收；舒适性差 |
| RGB运行 | 隔离worker + Xvfb驱动MetaDrive RGBCamera 128×64 CPU 5步 | `metadrive_camera_smoke.json` | sensor/runtime通过；未训练raw RGB策略 |
| 最终全量测试 | 在真实MetaDrive runtime执行单元+integration+coverage | `logs/v2_3/tests_with_coverage.log` | 178通过/1按设计跳过；src覆盖约91% |
| 发布收尾 | 更新当前README/验收/限制/模型索引，精选MetaDrive模型，单根ZIP+manifest+独立复验 | v2.3.0 verification JSON | 研究发行，不做安全认证声明 |

本轮没有把两次未完成的长SB3/单进程PPO尝试计为成功训练。所有正式宣称均以最终存在的 `result.json`、模型哈希和闭环评估为依据。

# v2.4.0 安全监督与真实 RGB 学习链路台账（2026-09-07 UTC）

| 内部阶段 | 实质动作 | 真实证据 | 判定 |
|---|---|---|---|
| 风险奖励 | reward v4 新增 TTC/closing/unsafe acceleration/emergency brake；保留旧版本兼容 | `test_v24_safety_rgb.py`；固定速度反例 | 目标函数实现通过；学习性能未得到稳定提升 |
| 安全系统 | 实现 stateful longitudinal supervisor、warning/critical/hysteresis/slew、逐步介入记录 | `src/safety/supervisor.py`；supervisor tests | 工程与行为单元通过 |
| 冻结安全评估 | 同一32k state PPO分别 raw / supervised；增加 release2/release3 独立 split | `reports/v2_4/release2_*`、`release3_*` | release3 raw 42.5/57.5；supervised 100/0，介入45.82%；只算系统结果 |
| RGB 可训练输入 | 接 MetaDrive RGBCamera → RGBEncoder → motion fusion → PPO actor/critic | `rgb_ego_384` experiment/model | 384真实world steps；encoder真实更新；驾驶能力失败 |
| RGB 跨观测蒸馏 | frozen state teacher实际控制环境，student记录RGB+motion并做actor MSE | `rgb_distilled_48_multiseed` | 48样本/6seed，MSE显著下降；新seed smoke timeout |
| 扩大RGB样本尝试 | 尝试收集更大多seed RGB数据 | `experiments/dev_smoke_v2_4_attempts`（发行排除） | 当前运行时采集超时；不计入正式完成 |
| 负结果处理 | reward v4 transfer/KL短实验未稳定超越source；不加入model registry | dev_smoke归档 | 不做“learned safety improvement”宣称 |
| 发布收尾 | 版本/README/API/验收/限制/模型注册、全量测试、wheel、manifest、独立解压复验 | v2.4 verification JSON | 由最终验证JSON确定 |

v2.4 的最重要结论不是“安全率100%”，而是**raw policy与外部安全系统的能力必须拆开**。当前 supervisor 在困难holdout近半时间介入，说明 learned policy 仍需继续训练；RGB链路真实可训练，但还不能驾驶。

# v2.5.0 Safety policy distillation 与规则依赖降低台账（2026-09-07 UTC）

| 内部阶段 | 实质动作 | 真实证据 | 判定 |
|---|---|---|---|
| 新指标与split | 新增release4/safety_train互斥seed域；介入修正幅度/严重度聚合指标 | `test_v25_safety_metrics.py`、split tests | 通过 |
| Safety distillation | 新增intervention-aware weighted actor distillation；无介入时禁止Adam数值漂移 | `src/safety/distillation.py`、unit tests | 通过；critic/log_std不训练 |
| 真实MetaDrive采集 | source PPO + supervisor在8000–8011真实运行；target为applied action | `experiments/v2_5/safety_distill_round1` | 1419 transitions / 733 interventions / 0采集碰撞 |
| 开发选择 | 同一collection dataset训练4个权重/epoch候选；只用validation 1000–1009 | `reports/v2_5/dev_*` | 4候选均10/10成功0碰撞；以速度优先选w0.75/e8 |
| release4冻结验收 | 候选冻结后首次运行7000–7039；base/distilled × raw/supervised四组paired eval | `reports/v2_5/release4` | raw collision57.5→40%；supervisor介入47.22→27.11% |
| 失败保留 | distilled raw在16/40场景碰撞；原始失败trace完整保留 | curated raw `failure_traces.json` | 明确未通过raw安全目标 |
| Waymo小回归 | distilled raw在assets 3个Waymo示例做ScenarioEnv真实闭环 | `reports/v2_5/waymo_distilled_curated` | 3/3到达；使用最终curated模型重跑，仅接口小样本，不做泛化声明 |
| 可视化证据 | 同seed raw success、raw collision、supervised recovery GIF；paired comparison图 | `videos/v2_5`、`reports/v2_5/*.png` | 成功与失败均展示 |
| 全量测试 | 真实MetaDrive runtime下unit/integration/coverage | `logs/v2_5/tests_with_coverage.log` | 207通过/1按设计跳过/1 Gym警告；src约90% |
| 发布收尾 | curated model provenance、registry、文档、版本、manifest、ZIP独立复验 | v2.5 verification JSON | 由最终复验结果决定 |

release4是在validation完成候选选择后才首次打开；本轮没有根据release4结果继续回选v2.5模型。后续任何优化必须新建未触碰split，避免把已查看release4继续当最终测试集。

## 2026-09-07 — v2.6.0 safety curriculum / DAgger / frozen-release cycle

| 内部阶段 | 实质动作 | 真实证据 | 判定 |
|---|---|---|---|
| 多阶段 correction learning | strict longitudinal、balanced encoder+actor、training-only lane teacher三条MetaDrive DAgger路线，分别使用互斥safety-training seeds | `experiments/v2_6/dagger_*` | 2,981/2,854/2,821 transitions；均保留负结果 |
| staged PPO | strict DAgger warm-start，三stage各1,024 MetaDrive world steps，reference KL约束到v2.5 source | `experiments/v2_6/ppo_curriculum_from_dagger` | 3,072 world steps完成；validation未晋升 |
| 评估可靠性 | 复现Panda3D teardown卡住；新增per-stage fresh process、`evaluation.complete.json` commit marker、request fingerprint及`--resume` | `scripts/evaluate_safety_suite.py` + tests | 完成结果先commit，再回收卡住worker；partial/fingerprint mismatch不复用 |
| 横向teacher探索 | 新增只读policy-visible lane/heading/route preview的训练期lane teacher；运行时SafetySupervisor仍不改steering | `src/safety/lane_teacher.py`、`dagger_lane` | 实现通过；validation不优于promotion门槛，保留为负实验 |
| 候选防泄漏选择 | 5个raw policy在相同validation 1000–1019上确定性排序；新增predeclared promotion gate | `reports/v2_6/candidate_selection.json` | `dagger_balanced`排名第一但promotion=false；curated registry不变 |
| release5冻结诊断 | 在候选冻结后首次评估9000–9039；base/selected × raw/supervised四组 | `reports/v2_6/release5_*`、`comparison.json` | raw collision -10pp但off-road +7.5pp；catastrophic仅-2.5pp；release5现已消费 |
| supervisor依赖诊断 | 相同longitudinal supervisor对比base/selected | `reports/v2_6/comparison.json` | 介入时间51.66%→23.61%，但success仍25%、off-road仍50%；横向/路径失败未解决 |
| 发布收尾 | 文档/验收矩阵/限制/报告、全量测试、wheel、manifest、单根ZIP独立解压复验 | v2.6 verification JSON | 以最终机器验证为准；不把未promote候选登记为推荐模型 |

关键方法学决策：release5从本轮冻结评估后永久视为已查看数据；后续任何调参必须建立新release6。模型ranking和promotion只允许validation seeds，release结果不得回流选模。v2.6最重要的结论是**纯纵向safety correction的边际收益已经明显，dense/curve/mixed地图需要独立的横向路径与交互能力课程**，而不是继续通过压低collision把失败转成off-road/timeout。

# v2.7.0 Privileged lateral expert / path diagnosis / release6 cycle（2026-09-08）

| 内部阶段 | 实质动作 | 真实证据 | 判定 |
|---|---|---|---|
| 新数据域 | 新增expert_train 10000–10999、validation2 11000–11199、release6 12000–12199；与历史域互斥 | split tests | 通过 |
| MetaDrive lateral expert | 内置IDMPolicy绑定ego，training-only privileged steering teacher | `src/experts/metadrive_idm.py` | 真实MetaDrive运行通过；部署不使用teacher |
| 正式expert采集 | steering-only 3,485 transitions/16seed；balanced 3,529/16seed | `experiments/v2_7/expert_steering`, `expert_balanced` | 7,014 transitions / 32互斥train seeds |
| Path suite | curve/mixed sparse/moderate四条件，支持同一longitudinal supervisor | `src/evaluation/path_suite.py`, reports/v2_7/path_* | 实测完成 |
| Anchored imitation | frozen source action + feature anchor；同时修复distillation每batch重复permutation问题 | `src/experts/distillation.py`, anchored models/tests | 机制通过；validation行为负结果 |
| Validation2选择 | base/steering/balanced/anchored在同一11000–11019 safety suite排名 | `selection_validation2.json` | balanced +5pp success恰好过promotion gate |
| Freeze | selected模型复制入models，固化model SHA与selection SHA | `reports/v2_7/FROZEN_MODEL.json` | release6打开前完成 |
| Release6主诊断 | 12000–12039，base/selected × raw/supervised四组真实MetaDrive闭环 | `reports/v2_7/release6_*`, `comparison.json` | raw未确认稳健安全改善；release6现已消费 |
| Release6 path诊断 | 20个curve/mixed secondary scenarios | `release6_path_*` | off-road 95→80%，仍严重失败 |
| 负实验保留 | anchored candidates与更大unconstrained DAgger不回选 | `experiments/v2_7/*` | 不通过行为目标，保留研究证据 |
| 发布收尾 | 文档、registry状态、全量测试、wheel、manifest、ZIP独立复验 | v2.7 verification | 以最终机器验证为准 |

关键结论：privileged lateral teacher能提供方向性的path-following信号并降低部分off-road，但release6 raw collision没有下降，有限validation promotion没有得到holdout确认。下一轮必须使用新split并进行multi-source-seed与interaction/path分层训练，不能继续查看release6调参。

# v2.8.0 Capability specialization / fresh release11 cycle（2026-09-08）

| 内部阶段 | 实质动作 | 真实证据 | 判定 |
|---|---|---|---|
| 能力拆分 | 新增curve/merge/bottleneck/intersection/dense五阶段MetaDrive capability suite | `src/evaluation/capability_suite.py` | 真实MetaDrive reset/eval通过 |
| 新seed域 | capability_train、validation3-7、release7-11全部与历史域互斥 | generator + v2.8 tests | 通过 |
| 负实验1 | anchored/balanced full-action DAgger | validation4 | success未改善/出现collision→off-road迁移；不晋升 |
| 负实验2 | component-wise imitation | validation5 | success +6.7pp但off-road +33.3pp；promotion=false |
| 负实验3 | reference-bounded component adaptation | validation6 | 0 success且completion下降；promotion=false |
| 最终采集 | pure-expert full-action IDM teacher，13500-13529 | `capability_pureexpert_v4` | 9,000 transitions / 30 unique seeds；teacher不部署 |
| Validation7 | source vs pureexpert，22000-22014 | `selection_validation7.json` | success 6.7→26.7%，catastrophic 93.3→73.3%；promotion=true |
| Chronology | model/selection mtime+SHA早于release11 requests；专门freeze JSON未提前创建 | `FREEZE_AUDIT.json` | hash/时间审计通过；流程瑕疵显式保留 |
| Release11 | 23000-23024 25 fresh capability scenarios | `comparison.json` | aggregate success 4→32%，collision 32→12%，off-road 64→56% |
| 能力分解 | bottleneck/intersection/dense改善；curve 100% off-road；ramp success 20→0% | release11 stage reports | aggregate改善确认，但非统一能力提升 |
| 配对统计 | 同stage/seed paired success | 8改善/1退步，exact McNemar p=0.0391 | 描述性研究证据，不扩大为安全结论 |
| 发布 | 版本/registry/docs/tests/manifest/ZIP独立复验 | v2.8 verification | 以最终机器验证为准 |

关键结论：v2.8首次在fresh capability holdout上确认了显著的aggregate能力提升，但curve/ramp仍是明显短板。release11已消费；v2.9必须使用新validation/release域，并应把拓扑/曲率表示和多source PPO seed作为重点，而不是继续复用release11调参。


# v2.9.0 Transparent hybrid route-control / release12 cycle（2026-09-08）

| 内部阶段 | 实质动作 | 真实证据 | 判定 |
|---|---|---|---|
| route表示 | 新增schema4按路径弧长排序的forward route preview；schema3保持默认兼容 | `test_v29_route_observation.py` | 表征/兼容性通过 |
| learned lateral实验 | schema4 pure-expert、steering weight4、blended DAgger、closed-form lateral-head ridge | `experiments/v2_9/*` | 离线拟合改善但raw curve/ramp闭环仍高出界；全部不晋升 |
| 不完整实验处理 | 大schema4 run执行超时，只产生3个stage checkpoint | `dev_partial_route_schema4_expert/STATUS.json` | 显式标记incomplete，不进入选择/发布 |
| 透明RouteAssist | pure-pursuit只改steering，并保留actor proposed action | `src/navigation/route_assist.py` + tests | 工程通过；明确是规则组件 |
| 纵横向分层 | RouteAssist后再进longitudinal SafetySupervisor；三层action全部写入info | MetaDrive integration + metrics | 控制权限分离通过 |
| Validation8 | raw / route / route+safety 在26000-26014同15场景 | `reports/v2_9/validation8_*` | hybrid 93.3% success/0 collision；选择系统配置，不选择新模型 |
| Pre-release freeze | 在release12任何stage request之前冻结model/config/core source hashes | `release12_freeze.json` | **真正pre-release freeze通过** |
| Release12 | 27000-27014 fresh 15场景，raw vs frozen hybrid | `reports/v2_9/release12_*`, `comparison.json` | raw 33.3%→hybrid93.3% success，catastrophic 66.7%→6.7% |
| 解释边界 | route介入99.82%、safety介入15.40%；system registry与model registry分离 | `models/system_registry.json` | 不把规则系统效果归为raw RL |
| Release治理 | release12标记consumed | `RELEASE_STATUS.json` | 后续必须validation9/release13 |

关键结论：v2.9证明透明neural+rule组合能在有限fresh MetaDrive capability holdout上显著提高系统完成率，并首次补齐真正的pre-release freeze流程；同时，接近100%的steering规则介入也明确证明 learned lateral policy 本身仍未解决，不能把system-level结果包装成RL策略能力。


# v3.0.0 Adaptive shared-control / release13 cycle（2026-09-08）

| 内部阶段 | 实质动作 | 真实证据 | 判定 |
|---|---|---|---|
| adaptive handoff | RouteAssist新增fixed/adaptive两模式；adaptive need只用policy-visible curvature/heading/lane features，max blend 0.5 | `src/navigation/route_assist.py` + v3 tests | 固定模式向后兼容；adaptive组件通过 |
| 指标审计 | info/metrics新增need、time-varying blend与learned steering authority | release13 reports | 可区分“规则介入频率”和“规则平均控制权” |
| 新数据域 | handoff_train / validation9 / release13与历史split互斥 | generator + split tests | 通过 |
| Validation9 | fixed与adaptive在29000–29014比较 | `reports/v3_0/validation9_*` | adaptive 93.3% success；mean blend 13.7%，通过预声明gate |
| Pre-release freeze | release13前冻结8个关键文件SHA与selection | `release13_freeze.json` | chronology/hash均通过 |
| Release13 | raw/fixed/adaptive在30000–30014 fresh 15场景 | `reports/v3_0/release13_*` | fixed/adaptive均80% success/20% off-road；adaptive mean blend 11.4% |
| trade-off | fixed→adaptive paired success 1改善/1退步，curve退步但intersection改善 | `comparison.json` | 不宣称success优于fixed |
| learned-model治理 | neural checkpoint保持v2.8，不新增v3 learned registry项；adaptive只进system registry | registries | 系统/模型结果分离 |
| 发布 | 文档、tests、wheel、manifest、ZIP独立复验 | v3.0 verification | 以最终机器验证为准 |

关键结论：v3.0在有限fresh holdout上把几何steering平均blend从100%降至11.4%，同时保持fixed系统相同的aggregate success/catastrophic，但仍有频繁小幅规则修正，且curve stage存在能力退步。它是**透明shared-control权限制衡结果，不是新learned policy性能提升**。release13已消费。

# v3.1.0 Learned Route Adapter / online DAgger / release14 cycle（2026-09-08）

| 内部阶段 | 实质动作 | 真实证据 | 判定 |
|---|---|---|---|
| 规则→学习模块 | 新增`LearnedRouteAdapter`，state+base action预测raw residual与learned blend；runtime不调用pure-pursuit | unit tests + artifact SHA | 通过 |
| 在线DAgger | adapter_train 31000–31014，三轮teacher mix=1.0/0.5/0.2，纵向采集时Supervisor防碰撞截断 | `experiments/v3_1/route_adapter_dagger` | 3,634 transitions /15 unique seeds |
| 指标审计 | evaluator注入agent diagnostics；新增adapter blend/correction/active fraction聚合 | metrics/tests | 通过 |
| Validation10 | base / v3.0 adaptive / adapter raw / adapter supervised，32000–32014 | `reports/v3_1/validation10_*` | adapter raw +13.3pp success、-13.3pp catastrophic，过promotion gate |
| Freeze | release14前冻结base/adapter/config/selection/core source SHA | `release14_freeze.json` | 通过 |
| Release14 | 33000–33014首次打开，base/adaptor/supervised/adaptive reference | `reports/v3_1/release14_*` | adapter raw 60% success/6.7% collision/33.3% off-road；4改善/0退步，p=0.125 |
| 负结果保留 | curve tracking仍0/3 success、3/3 off-road；adapter仍落后adaptive rule reference | `comparison.json` | 明确保留，不做“已解决route following”声明 |
| 注册治理 | learned adapter独立`route_adapter_registry.json`；带SafetySupervisor系统进入`system_registry.json` | registries | learned组件与rule系统结果分离 |
| 工程收尾 | 全量测试、real MetaDrive smoke、CLI/wheel/manifest/ZIP独立复验 | v3.1 verification | 以最终机器验证为准 |

关键结论：v3.1首次把部分RouteAssist能力转移到**runtime不调用几何controller的learned residual module**，并在fresh release14上获得方向性闭环改善；但curve仍彻底失败、样本量小、source actor单seed，因此这是能力转移研究证据，不是安全证明。release14已消费。

# v3.3.0-dev Real MetaDrive runtime / protocol cleanup（2026-09-09）

| 内部阶段 | 实质动作 | 真实证据 | 判定 |
|---|---|---|---|
| runtime恢复 | 接入用户上传的 MetaDrive 0.4.3 `assets.zip`，隔离安装 Panda3D/Gymnasium/pygame wheelhouse | `reports/v3_3/runtime_smoke.json` | source/assets version 均0.4.3；headless reset+64 step通过 |
| temporal闭环 | frozen v2.8 + identity v3.3 adapter 连续两episode | runtime smoke | state=62D；history 1→6；episode reset清零；action/diagnostics finite |
| mini DAgger | scene_train curve/ramp各1 seed，protected collection，2 epoch | `reports/v3_3/mini_dagger_smoke.json` | 200 transitions；15 supervisor interventions；artifact/base/dataset SHA绑定成功 |
| raw部署边界 | 重新加载mini adapter并关闭RouteAssist/SafetySupervisor | runtime smoke | runtime supervisor interventions=0；applied action等于agent action |
| MetaDrive回归 | 真实runtime执行 marker tests + 全量tests | `reports/v3_3/runtime_smoke.json` | MetaDrive 9/9；全量323 passed /1 skipped |
| representation协议修正 | 发现旧A/B/C/D把neighbor开关和risk auxiliary同时改变 | launcher/tests | 改为2×2 history×neighbors，A/B/C/D统一risk loss=0 |
| risk辅助独立消融 | 固定D representation，只比较R0=0与R15=.15 | `scripts/run_v33_risk_aux_ablation.py` | 与representation effect解耦；release16入口不存在 |
| validation治理 | 正式evaluator smoke使用validation12 seed38000，并用v3.2同seed做sanity baseline | `reports/v3_3/validation12_engineering_smoke.json` | validation12永久降级为engineering-consumed；selection迁移fresh validation13 40000–40199 |
| release治理 | 本轮从未运行release16 | launcher manifests + status | release16保持unopened，只有freeze后允许首次打开 |

关键方法学决策：validation holdout 一旦被工程 smoke 查看，也按“已消费”处理，而不是因为没有用于正式选模就继续当作 pristine validation。v3.3 后续正式比较因此只使用预注册的 `scene_train + validation13`；representation 与 risk auxiliary 两套消融都必须在首次打开 validation13 之前冻结实验矩阵。release16 继续保留为最终一次性 holdout。
