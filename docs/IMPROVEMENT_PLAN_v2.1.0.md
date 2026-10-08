# rl_autodrive_agent v2.0.0 阻塞解除与质量改进方案

状态：方案与针对性审计；不是修复完成报告，不修改已发布 ZIP，不覆盖既有实验结果。
依据：交付包代码、MetaDrive 安装失败日志、KNOWN_LIMITATIONS、验收矩阵，以及本次两个最小复核实验。

## 一、已确认的根因与边界

### 外部阻塞：MetaDrive 运行条件

既有日志中的实际错误是 DNS 解析失败（Temporary failure in name resolution）。后续“No matching distribution found”不能据此解释为包不存在或 Python 必然不兼容。历史诊断显示缺少 Gymnasium、Panda3D、pygame、progressbar、MetaDrive 与配套 assets；历史 MetaDrive 执行步数为零。本次不声称网络条件已有变化。

上传的源码归档不含 MetaDrive 运行所需的 metadrive/assets 目录。发布版脚本已有 --wheelhouse 和 --assets-zip 入参，需补齐来源/版本/ABI校验和分阶段诊断，而不是继续添加同类空接口。

### 内部设计问题 A：目标速度与即时奖励冲突

复核条件：native、直道、无背景车、目标速度12m/s、相同初始几何，油门只补偿阻力。12m/s时单步奖励约0.99817；24m/s时约1.89817。进度奖励的增长超过超速惩罚，因此当前即时奖励不能表达“达到目标速度后不再鼓励更快”。这不是对全部碰撞原因的因果证明，仍需闭环消融。

### 内部设计问题 B：单帧语义 BEV 缺少速度信息

保持车辆位置和航向不变，仅将自车速度从8改到24m/s，五通道BEV完全一致，状态向量发生变化。当前BEV-only前馈策略无法从该帧区分这两种状态。此问题不能仅靠增加训练步数消除。

原始复核结果：rl_autodrive_agent_improvement_audit.json。
可复核脚本：review_v2_observation_reward.py。它不是全量测试或修复脚本。

## 二、P0：解除真实 MetaDrive 阻塞

### 安装链路

固定上传源码的版本与内容SHA，不同时混用editable源码和另一个PyPI版本。为本项目创建独立环境；Python3.11可作为兼容性验证候选，不宣称未经测试的版本组合可用，也不把原Python3.13当成已证实根因。

线上分支：分别诊断DNS、HTTPS、依赖解析、动态库加载、assets下载与版本校验。
离线分支：在与目标OS/架构/Python ABI匹配的环境生成wheelhouse，包含传递依赖与必要构建依赖；缺轮子的包在匹配环境构建。CPU版PyTorch和解释器也必须匹配，不能搬用不同Python版本的site-packages。

安装产物应包含：源码SHA、assets版本与SHA、依赖锁、wheel清单、Python/OS/架构信息、pip check与导入结果。分开打包核心运行依赖、训练依赖、数据转换依赖，避免每次传输全部内容。

### 运行门禁

依次验证：真实导入；assets存在且版本匹配；关闭渲染的reset/step；实际终止/截断语义；100个种子的累计至少10,000个真实环境步；32,768步PPO训练；模型冷加载后100回合评估。数字仅定义工程冒烟门槛，不代表驾驶能力达标。

每次轨迹记录backend、实际环境类型、版本、场景seed、模型SHA和真实步数。任何MetaDrive阶段失败即报告失败，不静默切回native。Gymnasium checker和指定MetaDrive集成用例在该发布配置下不得skip。

先使用官方状态观测和默认奖励完成基准，再逐项接入自定义观测与奖励，避免一次修改所有变量。原生与MetaDrive不能因向量shape相同就假设动作、时间步、车辆状态和奖励语义相同；需校验控制频率、m/s单位、制动符号、航向和终态快照。RGB渲染作为独立门禁；headless不是软件渲染必然可用的保证。

优先修改：scripts/bootstrap_metadrive.py、scripts/doctor.py、requirements-metadrive.txt、src/envs/metadrive_wrapper.py及集成测试。

## 三、P0/P1：修复驾驶学习目标与观测

### 奖励

将进度项改成有界的路线进度收益；增加目标速度跟踪损失，明确超速不能靠更大进度奖励获益；使用前后车相对速度和同车道间距构建风险项，而非仅用欧氏距离。增加碰撞前数秒的动作、间距、闭合速度和奖励分解日志。惩罚的时间累积按实际dt定义，避免改变控制频率就改变权衡。

测试对：目标速度/超速、安全跟车/持续逼近、必要制动/无理由停车、安全换道/危险切入、成功/碰撞。先测试每项即时收益，再在可控短轨迹上比较累计收益。危险状态下停车不应被“低速惩罚”压过；不要以大碰撞惩罚或规则接管宣称安全保证。

### 观测

先做BEV+自车速度/加速度/航向角速度/导航目标；再做带ego-motion补偿的帧堆叠或CNN+GRU。单帧BEV-only保留作受限观测消融。进一步增加遮挡、量测丢失和延迟，不能以原始真值速度图冒充传感器估计。

测试：改变速度/接近速度应改变可用观测或时间序列；历史不跨reset；帧时间戳一致；padding/可见性mask正确；部署Actor不得读训练专用全局真值。

### 训练与课程

先以相同环境、动作、观测和奖励运行SB3 PPO参考，定位自研算法/任务设计/数据预算各自贡献。CPU优先小MLP与状态输入，比较1/2/4个采样进程并实测吞吐，不预先承诺并行越多越快。MetaDrive进程各自持有仿真器。

训练预算按32k冒烟、100k诊断、300k验证逐级扩大；仍在早期失败时分析原因，不盲目增加到百万步。记录KL、clip fraction、entropy、value loss、explained variance、梯度和动作饱和率。

可选IDM/PID示范的行为克隆预训练，然后PPO微调；专家数据成本单独记录并保留纯PPO对照。课程从短直道低速无车开始，再到弯道、跟车、慢车换道、急刹和多车；升级依据独立课程验证集，不用降低碰撞阈值制造晋级。

安全过滤器只作为独立系统对照：分别报告raw policy、policy+shield、纯规则，以及接管频率和任务完成率。

优先修改：src/envs/native_world.py、src/envs/observation.py、src/perception/、src/scenarios/curriculum.py、src/agents/trainer.py。

## 四、P1：真实多智能体质量

同一世界内先两车、后四车；IPPO与MAPPO固定场景、有效Agent采样预算和网络规模。增加汇入、让行和争用车道任务，避免仅在互不干扰车道上验证MARL。

集中式Critic可读全局训练状态；Actor保持局部可观测。测试Agent顺序置换、死亡/截断mask、终态bootstrap、无跨车轨迹污染与无全局状态泄漏。在此基础上消融value normalization、GRU和局部/团队奖励，不一次加入所有改动。

增加PettingZoo Parallel API适配和官方parallel_api_test、parallel_seed_test。正式统计以独立场景为聚类单位，同时报告每车成功、全车成功、任意碰撞、最差车辆完成度、等待/死锁和通行效率。

## 五、P1/P2：真实数据与原始传感器端到端

先完成一种数据集的最小授权子集，建议20～100个场景作为管线验收；经ScenarioNet转换后用ScenarioEnv实际加载。数据转换与训练依赖分环境固定。下载脚本支持来源记录、授权前提、断点续传、完整性校验与失败退出；不能绕过账号或许可。

场景验收包括坐标/单位、航向、时间间隔、有效帧mask、agent ID、尺寸、地图拓扑和初始化无重叠。按原始日志/场景划分训练验证测试，避免同一日志切片跨集合。

分别评估：原始轨迹回放；自车闭环+背景非响应回放；交互背景控制。不要把第二种称为完全交互闭环，也不要把替换IDM后的轨迹称为真实原始轨迹。

视觉分层：语义BEV控制基准；低分辨率RGB/深度+运动状态的实际传感器闭环；再考虑感知BEV重建。原始相机是项目视觉目标的额外门禁。软件渲染可以尝试但必须实测，不以关闭窗口等同于无图形依赖。世界模型不是传感器到控制端到端系统的必要中间模块，不增加无训练证据的占位目录。

## 六、P1：工程和评估补齐

| 项目 | 改进与门禁 |
|---|---|
| SAC恢复 | 保存actor、双Q、target Q、温度、所有优化器、replay数据/指针、RNG、归一化与计数；原子写入。验证固定回放批次的恢复更新一致。物理世界不能保存时明确回合边界恢复。 |
| 公平实验 | 同场景/奖励/观测/网络和预算；至少3个训练seed，正式候选每场景族200个固定测试场景；初筛少量验证场景。分别记录环境步、有效Agent步、优化次数和CPU耗时。 |
| 测试隔离 | 原20个测试场景已读过，降为开发回归集；新冻结holdout不用于调参。按训练seed和场景分别表达不确定性，零碰撞不能报告零风险。 |
| 舒适性 | 补横向加速度/jerk、转向率；使用实际dt，均值/RMS/p95/峰值并列；换道阶段单独分析，不能把合理变道全当偏离。 |
| 性能 | 分解物理、观测、推理、优化与渲染耗时；固定车辆数、进程数和图像尺寸；发布raw数据，不混用渲染FPS和训练吞吐。 |
| 环境/依赖 | 干净环境pip check通过；Gym和PettingZoo检查器实跑；MetaDrive必选测试不得skip。 |
| TensorBoard/Docker | 真实event reader加载事件；实际启动服务验证；Docker实际build/run并训练冒烟，不以配方存在作通过。 |
| 发布 | 按核心CPU、MetaDrive、传感器、数据等profile分别验收；目标profile缺关键依赖应失败；解压复验、源码/模型/数据SHA、失败轨迹与原始日志齐全。 |

建议项目内行为门槛（不是现有成绩或道路安全标准）：基础驾驶成功率>=95%、碰撞率<=2%；密集交通探索目标成功率>=70%、碰撞率<=10%。必须附样本量、场景定义、置信区间和规则对照；未达则继续记录失败，不修改为“完成”。

## 七、执行依赖关系

P0：MetaDrive安装与运行诊断，同时修复native奖励目标及BEV可观测性。
P1：单车可靠性、SB3参考、课程和实车数MARL；工程恢复与评估同步补齐。
P2：一种真实数据小闭环、原始相机输入、最后扩大场景与预算。

外部安装阻塞时，奖励、观测、课程、恢复、测试及评估工作仍可在现有native环境继续；不能将全项目停在pip重试。

## 参考依据

官方在线文档复核日期：2026-09-07。
MetaDrive：Installing MetaDrive、Training、pull_asset.py。
pip：pip download（目标平台、解释器与ABI）。
Gymnasium：Utility functions/check_env。
PettingZoo：Testing Environments/Parallel API Test。
Stable-Baselines3：RL Tips and Tricks、PPO、Vectorized Environments。
ScenarioNet：Waymo Example、Scenario Description与数据集文档。

本文提出的训练收益、场景通过率和兼容版本都需要后续实验证据，不构成效果承诺。
