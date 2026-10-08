# 算法实现与可解释边界

## PPO / IPPO / MAPPO

Actor 先把本车局部状态、语义 BEV 或两者编码为特征，再输出 2 维 Normal 均值与可学习 log_std。采样 latent z 后动作 a=tanh(z)。策略概率包含 tanh 的雅可比校正，使用稳定 softplus 形式，避免动作饱和处的数值问题。

PPO loss 使用 clipped surrogate、clipped value loss、Gaussian exploration entropy、梯度裁剪和近似 KL early stop。PPO 使用新 rollout，不复用 SAC 的 replay buffer。默认 gamma=0.99，GAE lambda=0.95，clip=0.2；这只是可复现配置，不声称是 MetaDrive 官方最佳超参数。[1]

TD residual：δ = r + γ(1−terminated)V(next) − V(current)。GAE 的递推跨步因子还要乘 (1−terminated)(1−truncated)。时间截断的 next value 来自 reset 前的真实末状态，而非下一回合首状态。 inactive-agent mask 在 rollout 与 loss 中均生效。[2]

参数共享 IPPO 用同一个局部 actor/critic 从多车经验训练；不是每车一份互不共享的网络。MAPPO 使用去中心化局部 actor，但 critic 输入所有学习车辆的 state、alive mask 和车辆 one-hot ID。推理阶段不读取该中心化状态。这里只实现这套 CTDE 变体，不声称完整复现原论文全部技巧，没有 RNN、PopArt 或 ValueNorm。[3]

## SAC

SAC 使用独立 replay、twin-Q、目标 Q 的 Polyak 更新、可学习 entropy temperature 和重参数化 squashed Gaussian actor。Replay 保留真正的 terminated，外部截断允许 bootstrap。图像保持 uint8 到取样后才转 float。当前仅支持单学习 ego，但背景车会响应交通流。[4]

v2.1 将推理 actor `model.pt` 与完整学习状态 `training_state.pt` 分开。后者保存双Q/target/温度/所有优化器/replay/RNG/计数，支持下一优化步逐张量一致恢复；物理世界未序列化，重新开始回合。旧actor-only无法冒充完整SAC恢复。

## BEV 与“端到端”

五通道是 drivable corridor、lane boundaries、route、other vehicles、ego，尺寸默认 48×48。三个卷积层、池化与线性层参与 PPO/MAPPO loss 反向传播。state/hybrid 分支使用 MLP/fusion。纯 BEV 策略不额外获取中心化状态或速度 state，MAPPO 的 privileged joint state 只供训练 critic 使用。

因此这里“端到端”的准确定义是 **模拟器语义输入到连续驾驶控制联合优化**。不是 raw camera-to-control，BEV 真值生成不是学习出来的感知模块，也没有 learned dynamics/world model。`result.json` 中 encoder_changed_l1 与测试证明 CNN 权重实际变化。

## Reward

旧版 reward_version=2 保留原公式用于精确复现，其中无界进度会压过超速惩罚。v2.1 新默认 reward_version=3 将进度截断至[-1,1]，增加目标速度跟踪与二次超速项、基于道路切向/相对运动的前后风险惩罚。密集奖励按 dt/0.1 缩放，碰撞/出界/到达为事件项，不随dt重复缩放。危险时下调安全目标速度，并避免舒适惩罚抵消必要制动。该设计修复固定反例，但短训练紧急场景实际退步，不能从公式推出策略已安全。

这些惩罚不是形式化安全约束。高奖励、低碰撞率或一段漂亮视频都不能保证安全。MetaDrive 保持 simulator 原奖励；跨后端 reward 数值不做直接比较。

## 单位与统计

速度 m/s，lane error m，jerk m/s³。时间加权的均值与 RMS 从实际 dt 计算。lane-crossing time 不是 off-road failure，合法换道也可能经过车道线。Agent episode 和 scenario episode 分开计数，多车同场景的统计不假定彼此独立。

参考：[1] PPO 原论文；[2] Gymnasium time limits 文档；[3] MAPPO 原论文；[4] SAC Algorithms and Applications。完整来源见 `references.md`。


## v2.1 时间观测与诊断

`bev_ego` 将语义图CNN与20维自车运动/导航编码融合；新多帧栅格先按自车姿态对齐、附帧龄与validity，再进入同一CNN。没有RNN，history堆叠不是learned world model。纯bev单帧特意保留作受限信息对照。

PPO增加可解释方差与其有效标志、梯度norm、动作饱和率等日志，不改变旧奖励模式的优化行为。显式reward_version=2复现历史三个种子，各32768步的9个权重张量完全一致；不能据此保证跨硬件的逐位复现。

风险计算是可审计启发式，对连续轨迹累计收益没有形式化安全证明。新reward的目标速度只影响学习目标，不是隐藏的规则接管；评估中的所有动作仍由神经策略决定。


## v2.5 Intervention-aware Safety Distillation

运行时 SafetySupervisor 不进入 neural actor；它只在真实 MetaDrive 中观察当前Snapshot，并把 actor 的 longitudinal command限制为 applied command。v2.5 将这些 `(state, proposed_action, applied_action, intervention, severity)` 转成离线监督样本，只更新 encoder+actor。

训练目标是 component-weighted MSE：透明样本权重为1，用于保持原策略；介入样本的sample weight随severity增加。steering target始终等于source actor steering，因为supervisor从不改方向。若一个数据集没有任何介入，优化器完全不运行，防止在数值舍入误差上产生Adam漂移。

该过程接近针对纵向安全修正的 DAgger-style imitation，但当前精选v2.5模型只使用一轮真实采集。标签来自启发式、privileged simulator risk teacher；不能理解为形式化safe RL。critic和log_std保持原值，因此蒸馏checkpoint用于actor部署和冻结评估，不是完整PPO续训状态。
