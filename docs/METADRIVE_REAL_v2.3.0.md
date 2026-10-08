# v2.3.0 MetaDrive 真实运行、训练与验收记录

## 1. 本轮解除的阻塞

用户提供了与沙盒目标环境匹配的 `metadrive_wheelhouse.zip`。经清单、wheel tag、哈希与依赖关系检查后，本轮在隔离 overlay runtime 中安装并实际导入：Gymnasium、Panda3D、pygame、progressbar、panda3d-gltf/simplepbr、mediapy、yapf、PettingZoo、Stable-Baselines3、TensorBoard 等依赖；MetaDrive 源码与 assets 均为 **0.4.3**。

安装完成后真实 MetaDrive `reset/step` smoke 执行 40 步通过。详细机器证据：

- `reports/v2_3/runtime_creation.json`
- `reports/v2_3/metadrive_installation.json`
- `reports/v2_3/metadrive_runtime_smoke.json`

全局 Python 环境仍存在与项目无关的 `moviepy -> Pillow<12` 预装冲突。本项目依赖闭包通过，且使用独立 overlay；没有把“全局 pip check 有冲突”改写成“全局环境完全干净”。

## 2. Gymnasium 兼容修复

MetaDrive 0.4.3 的原始 `BaseEnv.reset` 不接受 Gymnasium 1.2 checker 传入的 `options`。新增 `OfficialMetaDriveGymEnv` facade：

1. 暴露标准 `reset(seed=None, options=None)`；
2. 将 Gymnasium seed 映射到配置的 scenario seed 区间；
3. 为官方 SB3 参考链路提供标准 Env/Space；
4. 非空、未实现的 `options` 明确拒绝，不静默忽略；
5. checker 关闭 registry-based close/re-make 检查，因为 MetaDrive 的 Panda3D engine 生命周期不是 Gym registry 环境。

真实安装环境的测试使 MetaDrive/Gymnasium 集成测试从历史的“缺依赖跳过”转为执行通过。

## 3. 真实训练证据

### 3.1 官方 observation/reward + SB3 PPO

`experiments/v2_3/metadrive_official_ppo_smoke4096/result.json`

- SB3 PPO 实际 timesteps：4,096；
- 训练+冷评估 MetaDrive steps：4,343；
- 参数 L1 变化：261.08；
- 10 回合冷评估：success 0%，collision 0%，out-of-road 100%。

这证明官方参考训练链路真正运行，但短预算策略能力很差。更长的单进程尝试曾发生晚期停滞，因此没有把不完整的 32k SB3 尝试计为完成实验。

### 3.2 项目 PPO state

真实 MetaDrive PPO 通过检查点续训到 **32,768 世界步**。由于单个长期 Panda3D 进程出现过不可预测停滞，最终链路由 8,192 步检查点起按新进程续训完成。训练状态连续，仿真物理 episode 不跨进程恢复。

证据：

- `reports/v2_3/training_chain_32768.json`
- `experiments/v2_3/custom_metadrive_ppo_state_resume_24576_to_32768/result.json`
- `models/metadrive_ppo_state_32768.pt`

固定 20 个 held-out 场景闭环评估：

- success：75%；
- collision：25%；
- out-of-road：0%；
- mean route completion：0.923；
- average speed：11.30 m/s；
- inference p95：约 0.40 ms。

该碰撞率仍然过高，模型不能描述为“安全驾驶策略”。

### 3.3 真实多智能体 MAPPO

两台学习车辆在**同一个 MetaDrive 物理世界**中训练，使用 CTDE MAPPO 与 `hybrid` 输入。正式实验 8,192 世界步、13,429 个有效 agent steps。

20 个 held-out 场景 / 40 agent episodes：

- per-agent success：82.5%；
- joint success：80%；
- per-agent collision：17.5%；
- scenario-any-collision：20%；
- out-of-road：0%。

证据：`reports/v2_3/custom_metadrive_mappo_hybrid_8192_eval/`。

### 3.4 MetaDrive 语义 BEV

`bev_ego` 四帧 CNN PPO 在 MetaDrive 上训练 4,096 步，模型与 encoder 参数均发生非零变化，证明视觉 encoder 在真实优化图中。

10 个测试场景全部出界：success 0%、collision 0%、out-of-road 100%。因此当前结果只验收**链路和梯度**，不验收 BEV 驾驶能力。

## 4. 分段训练机制

新增 `scripts/train_metadrive_segmented.py`，每段使用独立子进程，超时后从最后接受的模型恢复。实际 8,192 世界步 / 2×4,096 段实验通过：`experiments/v2_3/segmented_metadrive_ppo_state_8192/result.json`。

分段恢复内容：模型、优化器和训练计数器。段边界仿真器重新 reset，因此不是物理状态/随机事件逐比特恢复。该机制解决的是长进程工程可靠性，不提高算法能力本身。

## 5. ScenarioEnv / Waymo 示例

MetaDrive assets 中存在 3 个已转换 Waymo 示例。v2.2 已完成静态轨迹审计；v2.3 进一步使用 `ScenarioEnv`，让冻结的 32k PPO 控制 ego 闭环运行全部 3 个场景。

结果：3/3 success，0 collision，0 out-of-road；但只有三个样本，Wilson 区间很宽，而且舒适性指标较差（例如 lateral jerk RMS 约 17.06 m/s³）。只能说明数据到仿真的闭环接口可运行，不能说明真实数据泛化能力。

## 6. RGB 相机运行

直接使用不可靠 DISPLAY 的 Panda3D 图像渲染可能导致底层进程崩溃。`scripts/metadrive_camera_smoke.py` 将渲染放在隔离 worker 中，并优先使用 `xvfb-run`。

当前 CPU 实测：128×64、stack=1、5 step，首帧 `[64, 128, 3, 1]`，平均 step 约 0.164 s，报告 `reports/v2_3/metadrive_camera_smoke.json`。

这只证明 **MetaDrive RGBCamera → CPU ndarray** 可用。当前策略没有以 raw RGB 为输入完成训练，因此不能宣称相机端到端策略已经实现。

## 7. 当前建议下一步

1. 把 32k state PPO 扩展到至少 3 个训练种子，并在冻结测试集统计跨种子分布；
2. 针对 collision cases 分析 TTC/前车风险并继续修复紧急制动；
3. 为 BEV 策略增加更长训练与课程学习，证明视觉输入不只是能反传；
4. 将 raw RGB 接到轻量 CNN policy 做最小可训练闭环，再讨论真正视觉端到端；
5. 增加更多授权 Waymo/ScenarioNet 场景，按 log 切分训练/验证/测试；
6. 多智能体加入 merge/yield/争用场景，而不是只提高算法复杂度。
