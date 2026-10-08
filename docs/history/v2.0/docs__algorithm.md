# 算法实现与可解释边界

## PPO / IPPO / MAPPO

Actor 先把本车局部状态、语义 BEV 或两者编码为特征，再输出 2 维 Normal 均值与可学习 log_std。采样 latent z 后动作 a=tanh(z)。策略概率包含 tanh 的雅可比校正，使用稳定 softplus 形式，避免动作饱和处的数值问题。

PPO loss 使用 clipped surrogate、clipped value loss、Gaussian exploration entropy、梯度裁剪和近似 KL early stop。PPO 使用新 rollout，不复用 SAC 的 replay buffer。默认 gamma=0.99，GAE lambda=0.95，clip=0.2；这只是可复现配置，不声称是 MetaDrive 官方最佳超参数。[1]

TD residual：δ = r + γ(1−terminated)V(next) − V(current)。GAE 的递推跨步因子还要乘 (1−terminated)(1−truncated)。时间截断的 next value 来自 reset 前的真实末状态，而非下一回合首状态。 inactive-agent mask 在 rollout 与 loss 中均生效。[2]

参数共享 IPPO 用同一个局部 actor/critic 从多车经验训练；不是每车一份互不共享的网络。MAPPO 使用去中心化局部 actor，但 critic 输入所有学习车辆的 state、alive mask 和车辆 one-hot ID。推理阶段不读取该中心化状态。这里只实现这套 CTDE 变体，不声称完整复现原论文全部技巧，没有 RNN、PopArt 或 ValueNorm。[3]

## SAC

SAC 使用独立 replay、twin-Q、目标 Q 的 Polyak 更新、可学习 entropy temperature 和重参数化 squashed Gaussian actor。Replay 保留真正的 terminated，外部截断允许 bootstrap。图像保持 uint8 到取样后才转 float。当前仅支持单学习 ego，但背景车会响应交通流。[4]

SAC final checkpoint 明确为 actor 推理权重；没有把缺失 replay/目标网络/温度优化器的载入称为完整恢复。PPO/MAPPO 可以恢复优化器与计数器，但不包含模拟器逐帧快照。

## BEV 与“端到端”

五通道是 drivable corridor、lane boundaries、route、other vehicles、ego，尺寸默认 48×48。三个卷积层、池化与线性层参与 PPO/MAPPO loss 反向传播。state/hybrid 分支使用 MLP/fusion。纯 BEV 策略不额外获取中心化状态或速度 state，MAPPO 的 privileged joint state 只供训练 critic 使用。

因此这里“端到端”的准确定义是 **模拟器语义输入到连续驾驶控制联合优化**。不是 raw camera-to-control，BEV 真值生成不是学习出来的感知模块，也没有 learned dynamics/world model。`result.json` 中 encoder_changed_l1 与测试证明 CNN 权重实际变化。

## Reward

Native reward 是归一化前向进度，减去车道偏移、jerk 与超速项，加终点奖励并扣除碰撞/出界惩罚。Reward 会偏好更多前进，而不是静止在车道中心刷正奖励。车道奖励基于最近车道，换道过渡期会有代价但不是禁止换道。

这些惩罚不是形式化安全约束。高奖励、低碰撞率或一段漂亮视频都不能保证安全。MetaDrive 保持 simulator 原奖励；跨后端 reward 数值不做直接比较。

## 单位与统计

速度 m/s，lane error m，jerk m/s³。时间加权的均值与 RMS 从实际 dt 计算。lane-crossing time 不是 off-road failure，合法换道也可能经过车道线。Agent episode 和 scenario episode 分开计数，多车同场景的统计不假定彼此独立。

参考：[1] PPO 原论文；[2] Gymnasium time limits 文档；[3] MAPPO 原论文；[4] SAC Algorithms and Applications。完整来源见 `references.md`。
