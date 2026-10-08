# v2.1.0 需求—实现—验收矩阵

版本号与验收状态分开：CPU 工程改进已交付，MetaDrive/真实视觉/驾驶安全仍不完整。

| 需求/改进项 | 当前实现 | 验收证据 | 判定 |
|---|---|---|---|
| FR1 native闭环/动力学/混合交通 | 同一世界自行车模型和 IDM；实际训练多车 | experiments/v2_1；tests | native通过，非高保真实车 |
| FR1 MetaDrive真实部署 | 不降级适配器、source/assets版本门禁、wheel哈希/ABI与网络诊断 | reports/v2_1/metadrive_preflight.json；metadrive_doctor.json | **外部阻塞，0真实step** |
| FR2 PPO/SAC/MARL | 自研 PPO/共享IPPO/CTDE MAPPO；新语义运动输入训练 | matrix_results.json；新模型 | CPU运行验证通过，非全论文复现 |
| FR2 奖励修复 | version3有界进度、目标速度、时距/TTC/横向交互风险、dt权重 | improvement_audit.json；三种子对照 | 特定缺陷修复，紧急制动退步 |
| FR2 可观测性 | 20维运动/导航、1–8帧BEV、ego补偿、时间/有效性 | test_v21_observation.py；真实CNN变化 | 新模式通过；旧纯BEV保留缺陷 |
| FR2 SAC完整续训 | 双Q/target/温度/优化器/replay/RNG/计数 | test_v21_sac_resume.py；2048+2048实际续训 | 学习状态通过；无物理快照 |
| FR3 闭环评估 | 1100场景-策略回合、1120 agent回合、配对/场景级统计 | reports/v2_1/evaluations | 开发验证，不是1100独立场景 |
| FR3 舒适性/失败分析 | 纵横向jerk、横加速度、转向率、末4秒失败记录 | test_v21_metrics.py；failure_traces.json | 通过；不等于舒适行业标准 |
| FR4 程序场景 | 交通密度、道路曲率、跟车/紧急场景 | dense/emergency实际评估 | 部分道路类型；无完整信号灯/环岛 |
| FR4 真实数据 | 通用授权下载/校验/CSV/回放；无新真实集 | 历史数据测试 | 真实nuScenes/Waymo未验收 |
| 可复现性 | 显式旧奖励；3种子每个32768步重现9个权重张量 | legacy_reproduction.json | 当前CPU运行时通过；非跨硬件担保 |
| 全量测试 | 127通过，5跳过，源码覆盖率见RELEASE_STATUS | pytest.xml；coverage.json | 工程回归通过；跳过不算通过 |
| 性能 | state约561step/s；四帧bev_ego约306step/s | performance.json | 满足该配置100step/s；未全面500 |
| 工程交付 | README/API/台账/模型/原始报告/清单/ZIP/独立解压复验 | MANIFEST；ZIP旁verification | 由最终独立验证报告确定 |
| 生产端到端驾驶 | 语义输入到控制实测；无原始相机感知 | KNOWN_LIMITATIONS | **未全部完成，禁止实车使用** |

先前改进方案中的独立课程重构、专家模仿预训练、安全过滤器、PettingZoo标准测试、真实数据转换、原始RGB和干净Docker环境本轮未完成，不能列作已交付功能。
