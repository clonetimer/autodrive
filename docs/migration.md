# 从旧归档迁移

上传的 v1.0.0 包包含两个不一致根目录、简化环境以及算法/适配占位接口。它没有可恢复的真实训练权重，因此本轮不把“之前已经完成”作为验收证据。原始 ZIP 不修改，审计记录在 `INPUT_AUDIT.json`。

v2.0 保留 `src/envs`、`src/agents`、`src/scenarios` 等模块布局，但以下变更不兼容旧占位代码：

- 单车 step 使用 `(observation, reward, terminated, truncated, info)`，多车使用对应五个按 Agent ID 索引的 dict，不能再假定四返回值。
- `SimpleDriveEnv` 包装真实耦合原生交通世界；多车必须通过 `make_parallel_env(EnvConfig(num_agents=N))`，不能复制 N 个单车环境假称 MARL。
- 观测为 `20+7*N` 维局部状态、5通道BEV或二者的 dict。2维动作是归一化转向和**有符号纵向踏板**，负值代表制动。
- 算法改为可在沙盒现有 PyTorch 上实际训练的本地 PPO/IPPO/MAPPO/SAC，不是 SB3 包的一层空调用。模型为版本2 `.pt`，不兼容 SB3 `.zip`；`--resume` 不接受旧占位文件。
- Eval 的未到终点超时不是成功；出界和碰撞分开统计，旧 `comfort_score` 改为有物理单位的 jerk。
- 配置统一 dataclass/YAML，可组合 include，并非已安装 Hydra。运行目录必须新建或显式 resume。
- MetaDrive 未满足依赖时抛异常并写失败证据，不再悄悄返回 SimpleDrive。

CLI 安装与训练命令见 README。项目模块在包名 `src` 下，这不是推荐给所有 Python 项目的通用命名，但保留了此次项目的用户目录约定。
