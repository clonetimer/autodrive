## v3.3.0-dev — Scene-aware temporal residual foundation

- 新增 `SceneAwareTemporalAdapter` 与 `SceneEncoder`：temporal route GRU + masked neighbor set attention。
- 新增 diagnostic-only safety-risk auxiliary head；训练可用 Supervisor label，raw runtime 不调用 Supervisor、risk head 不接管动作。
- 公共 route-adapter loader 支持 `scene_aware_temporal_adapter_v3`，并加入有边界的 episode history。
- 修复 evaluator neural warm-up 对 temporal agent 的历史污染：warm-up 后强制 reset。
- 新增 `train_scene_aware_adapter.py`，DAgger transition schema 扩展为 teacher/learner/executed/applied actions、step、reward、risk/failure labels。
- 新增 `scene_train/validation12/release16` seed domains；真实 evaluator smoke 后 validation12 seed 38000 被工程性消费，因此正式 selection 迁移到 fresh `validation13=40000–40199`。release16 仍未打开。
- 用户提供的 MetaDrive 0.4.3 assets 已接入隔离 runtime；headless/project temporal/mini-DAgger/raw-deployment smoke 均通过，全量回归 323 passed / 1 skipped（含 9 个真实 MetaDrive marker tests）。
- 当前状态明确为 development candidate，不包含新的 MetaDrive promotion/release 性能结论。
- `train_scene_aware_adapter.py` 新增显式 `--training-seed` 与严格 `run_config.json` fingerprint；resume 参数不一致时拒绝续跑。
- 修正 representation ablation 混杂项：A/B/C/D 现在只改变 history/neighbors，四组统一 risk-loss=0；新增独立 `run_v33_risk_aux_ablation.py` 比较固定 D representation 下的 R0/R15。
- scene_train-only 二轮 DAgger pilot 中 D 的 iteration-1 diagnostic steering MSE 最低，但四组均有 1/4 off-road；R15 相比 D/R0 恶化 steering fit，因此默认 `train_scene_aware_adapter.py --risk-loss-weight` 改为 0.0，辅助风险监督仅显式 opt-in。
- 两个 v3.3 launcher 都只允许 scene_train/validation13，暴露 bounded worker timeout/collection-step controls，且不提供 release16 入口。
- representation/risk launcher 新增 job-aware resume + plan fingerprint：矩阵中 fresh/partial/completed trial 可安全区分；真实 MetaDrive smoke 已验证“新目录+--resume”可启动、第二次会跳过已完成 job。

## v3.2.0 — Curve-aware direct residual adapter / fresh release15

- 新增 `CurveRouteAdapter`：policy-visible route-intent encoder + direct steering residual，release runtime 不调用 pure-pursuit。
- 真实 MetaDrive `curve_train` 34000–34011 三轮 DAgger，共 4,417 transitions / 12 unique seeds。
- 新增 validation11/release15、预声明 promotion gate、release15 前 freeze manifest 与冻结 registry 快照。
- validation11：v3.1 raw 60% success/40% catastrophic → v3.2 100%/0%，候选通过 gate。
- fresh release15：v3.1 raw → v3.2 raw 为 60.0%→93.3% success、13.3%→0% collision、26.7%→6.7% off-road；paired 5改善/0退步，McNemar p=0.0625。
- curve tracking 从 0/3 提升至 2/3，但仍有 1 个 off-road；不宣称安全或完全解决曲线驾驶。
- 修复 curve/ramp DAgger worker 可恢复执行与 direct-residual 模型统一加载。

## v3.1.0 — Learned Route Adapter / online DAgger transfer

- 新增 `LearnedRouteAdapter`：将 state + frozen base action 映射为 learned residual steering 与 blend，运行时不调用 pure-pursuit。
- 在真实 MetaDrive `adapter_train` 31000–31014 上运行三轮 online DAgger，采集 3,634 transitions；teacher mix 依次 1.0/0.5/0.2。
- 新增 validation10/release14、防泄漏 selection gate、pre-release freeze 与 immutable source snapshots。
- fresh release14 上 raw base → learned adapter：33.3%→60.0% success、26.7%→6.7% collision、40.0%→33.3% off-road；curve tracking 仍 0/3。
- 将 adapter-aware evaluation 拆成 v3.1 专用模块，历史 v3.0 freeze 继续校验冻结源码快照而非要求当前源码永久不变。
- 新增 route-adapter registry、CLI、真实 MetaDrive integration tests 与发布一致性测试。

# Changelog

## v3.0.0 — Adaptive shared-control authority reduction and release13 freeze

保持冻结v2.8 neural checkpoint，不晋升新learned model。RouteAssist从v2.9固定100% pure-pursuit steering替换为adaptive shared control：need score只读取policy-visible curvature/heading/lane features，几何转向blend限制在0～0.5。validation9选择max-blend=0.5；随后在release13之前真正冻结model/config/core source/selection SHA。fresh release13 15场景中fixed与adaptive均为80% success/0% collision/20% off-road，但adaptive mean geometric blend仅11.4%，mean learned steering authority 88.6%。fixed→adaptive paired success 1改善/1退步，McNemar p=1.0，route completion也从92.9%降至84.5%，curve stage更差，因此不宣称success提升。系统仍在76.5%时间步产生非零RouteAssist修正，SafetySupervisor约14.1%时间步修改纵向动作；结果严格归类为transparent shared-control system，非raw RL性能或安全认证。validation9/release13已消费。

## v2.9.0 — Transparent hybrid route control, schema-v4 and true pre-release freeze

新增 arc-length route observation schema v4、policy-visible pure-pursuit RouteAssist、route/safety intervention 分离指标、lateral-head distillation和真实MetaDrive contract test。多条schema4/DAgger/ridge learned-lateral路线在validation8闭环未稳定解决curve/ramp，因此没有晋升新learned checkpoint。透明hybrid在validation8为93.3% success/0% collision/6.7% off-road，并在**提前写入freeze manifest**后首次打开release12：raw 33.3% success/13.3% collision/53.3% off-road，对比hybrid 93.3%/0%/6.7%，paired success 10改善/1退步（exact McNemar p=0.0117）。但RouteAssist介入99.82%、SafetySupervisor介入15.40%，结果明确归类为hybrid neural+rule system，不归因于raw RL。release12已消费。

## v2.7.0 — Privileged lateral expert, path suite and release6 freeze

新增MetaDrive `IDMPolicy`训练期特权横向teacher、`expert_train/validation2/release6`新隔离seed域、curve/mixed path suite、anchored expert distillation负实验，以及validation2-only选择/冻结后release6诊断。正式两条expert路线共采集7,014条真实MetaDrive transitions/32个互斥训练seed。`expert_balanced`在20场景validation2恰好以+5pp success通过工程promotion gate并进入registry，但40场景release6未确认稳健安全改善：raw success 15→17.5%、collision 77.5→80%、off-road 7.5→2.5%，catastrophic仅85→82.5%，paired success 2改善/1退步，McNemar p=1.0。相同longitudinal supervisor下介入51.47→25.36%，但success 25→22.5%、collision 0→7.5%。secondary path diagnostic显示off-road 95→80%和route completion 25.97→36.35%，仍远未通过横向能力目标。release6现已消费。


## v2.6.0 — Safety curriculum, frozen selection gate and resumable MetaDrive evaluation

新增三条真实MetaDrive correction-learning路线（strict longitudinal DAgger、balanced encoder+actor DAgger、training-only lane teacher）、staged PPO、四条件safety suite、`evaluation.complete.json` commit marker、fresh-process/resume评估、validation-only确定性候选排序与预声明promotion gate。

候选选择只使用validation 1000–1019。`dagger_balanced` 排名第一，但相对base success +0pp、catastrophic -5pp、off-road +20pp，未通过promotion gate，因此不进入curated registry。模型冻结后首次打开release5 9000–9039：raw success 22.5→25%、collision 75→65%、off-road 2.5→10%，collision-or-off-road仅77.5→75%；paired success 1改善/0退步，exact McNemar p=1.0。相同longitudinal supervisor下介入时间51.66→23.61%，但success仍25%、off-road仍50%，暴露dense/curve/mixed场景的横向/路径失败。release5现已消费，未来调参必须新建release6。

# 变更记录

## v2.5.0 — Intervention-aware safety policy distillation

新增真实 MetaDrive SafetySupervisor correction dataset、intervention-weighted actor distillation、release4 冻结 paired evaluation、介入幅度/严重度指标和 raw-vs-supervised 对照演示。固定32k PPO经蒸馏后在 release4 40场景 raw success 42.5%→60.0%、collision 57.5%→40.0%；组合系统仍为100%/0%，supervisor介入时间47.22%→27.11%。这说明规则依赖下降但 raw policy 仍不安全。最终模型由 validation 1000–1009 选择，release4 在模型冻结后才评估。

## v2.4.0 — Auditable safety supervision and trainable real-RGB path

新增 reward v4（TTC/closing/unsafe acceleration/emergency brake shaping）、独立有状态纵向 SafetySupervisor、第三冻结 holdout、真实 MetaDrive `rgb_ego` CNN policy、跨观测 privileged-state teacher 行为克隆以及更严格的 raw-policy / supervised-system 分离报告。固定32k state PPO在release3 40场景为42.5%成功/57.5%碰撞；加supervisor为100%/0%，但介入时间45.82%，因此只证明系统层过滤有效，不证明raw RL安全。真实RGB PPO训练384步并确认CNN参数更新，但smoke几乎停驶；48样本多seed蒸馏将MSE从0.37147降至0.000150，闭环仍timeout。短reward-v4迁移实验没有稳定优于原策略，不进入精选模型。

## v2.3.0 — Real MetaDrive training, MARL, ScenarioEnv and RGB runtime

解除MetaDrive 0.4.3离线依赖阻塞：真实Gymnasium/MetaDrive集成测试、官方SB3 PPO参考训练、自研PPO 32768步、两车MAPPO 8192步、MetaDrive四帧BEV PPO、3个Waymo示例ScenarioEnv闭环以及CPU/Xvfb RGBCamera均实际运行。新增分段训练以缓解长Panda3D进程偶发停滞。最终MetaDrive PPO测试成功75%/碰撞25%，MAPPO per-agent成功82.5%/scenario-any-collision20%；BEV短训练10/10出界，均不作为安全声明。


## v2.2.0 — MetaDrive offline deployment engineering release

真实部署、CRC/SHA-256/版本校验 152 个 0.4.3 assets 文件，验证 7 处 glTF 引用；在新增 CPU 覆盖环境完成真实源码的 editable 安装。上传安装 ZIP 只有 4 个说明/脚本文件，没有 wheels；真实 MetaDrive 步数仍为 0。

新增安全可重复 staging、目标版本/ABI 校验、跨平台缺失依赖收集工具、官方观测与奖励的 CPU SB3 参考入口。修复 ScenarioEnv 配置键、行人碰撞记录、转向单位及横向舒适性统计。审计既有 assets 中 3 个 Waymo 示例（每个91帧），不是仿真回放。

本轮全量测试169通过/5跳过，src行覆盖86.59%；实际执行 native CLI 训练/评估/恢复回归，22个历史模型哈希/加载/短闭环验证。没有新增正式RL实验，不覆盖v2.1的负结果。下载工具网络端到端尚未验收。

## v2.1.0 — Reward, temporal BEV and SAC state recovery

有界进度与目标速度风险奖励、BEV+motion/时间堆叠、自车运动补偿、完整 SAC 学习状态恢复、物理舒适性评估与失败片段诊断。真实CPU训练241664世界步，10次新训练和1段续训；127通过/5跳过，src覆盖86.13%。密集交通改善但紧急制动退步，详细历史报告见 `IMPROVEMENT_REPORT_v2.1.0.md`。

## v2.0.0 — CPU research preview

**版本号不表示所有目标验收通过。** 此版本替换之前 ZIP 的占位实现，交付实际原生CPU实验与证据。

### 已运行

实现共享物理世界与IDM交通、真实PPO/SAC、参数共享IPPO/CTDE MAPPO、可训练语义BEV CNN、hybrid encoder、闭环部署、物理评估和场景区间、授权下载及轨迹回放、联合回合课程、配置组合、实验管理与完整性验证。

CPU正式实验共11组245760世界步；4096步复跑与早期存档权重逐张量一致。保留负结果：规则基线在简单测试中优于RL、密集交通和MARL碰撞仍多、课程未晋级。

### v1.1目标仍阻塞

MetaDrive真实适配器已完成代码接入和官方/上传源码对照，但沙盒缺少二进制依赖/assets且下载失败；未执行真实MetaDrive步，不能称为v1.1真实训练完成。

### 兼容变化

旧占位版 `.zip` 模型接口不兼容；所有本地算法使用版本2 `.pt`。step为Gymnasium五返回语义。完整迁移说明见 `migration.md`。

## v2.8.0 — Capability specialization and release11 holdout

- Added five-stage MetaDrive capability suite: curve tracking, ramp merge, bottleneck yield, intersection yield and dense progress.
- Added fresh capability seed domains through validation7/release11 and tests that enforce split disjointness.
- Added full-action privileged MetaDrive IDM teacher DAgger/behavior-cloning workflow plus anchored, component-wise and reference-bounded negative baselines.
- Preserved failed validation4/5/6 screens instead of repeatedly tuning on the same validation domain.
- Final pure-expert collection used 30 training seeds and 9,000 transitions with 100% expert action execution; deployment remains the neural student only.
- validation7 promoted the final candidate; fresh release11 aggregate improved success 4%→32%, collision 32%→12%, off-road 64%→56%, while curve/ramp stage failures remain explicit.
- Added capability selection/promotion logic, release11 chronology/hash audit, paired exact McNemar diagnostics and v2.8 report generation.
- Added curated v2.8 capability model with SHA-256 provenance; still explicitly not safety certified.
