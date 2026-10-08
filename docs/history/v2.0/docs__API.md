# API 约定

## 环境

`NativeWorld(EnvConfig)` / `MetaDriveParallelEnv(EnvConfig)` / `ReplayWorld(EnvConfig)`：

```python
observations, infos = env.reset(seed=2000)
while env.agents:
    actions = {name: policy.predict(observations[name]) for name in env.agents}
    next_obs, rewards, terminated, truncated, infos = env.step(actions)
    observations = {name: next_obs[name] for name in env.agents}
env.close()
```

`reset` 返回字典映射；`step` 返回五项字典，不使用 `__all__` 伪 agent。`possible_agents` 固定，`agents` 动态缩减。终态仍在该 step 的返回值中。下一 step 不得再给已终止车辆动作。无活跃车辆后必须 reset。

`SimpleDriveEnv` / `MetaDriveWrapper` 是单车五返回值 facade。有真实 Gymnasium 时继承其 Env；没有时仅适用于本项目 native trainer，不声称 Gymnasium 注册兼容。

## 观测 schema=2

- state：float32，长度 `20 + 7 * neighbors`，默认 62。
- bev：`{"bev": uint8[5,H,W]}`。
- hybrid：`{"state": float32[62], "bev": uint8[5,H,W]}`。

state 前 12 项：速度/30、航向误差 sin/cos、最近车道偏移/宽度、加速度/6、目标速度/30、道路横向偏移归一化、车道数/4、车道宽/5、剩余完成度、转向/0.45、车道序号归一化。

随后 4 个 route 点：局部 x/60、y/20。每个 neighbor 7 项：局部 x/60、y/20、相对 vx/30、vy/30、相对 heading sin/cos、presence mask。邻车按距离与 ID 排序，缺失项补零，观测噪声不污染 padding mask。

Actor 没有绝对世界坐标输入，但路线、交通真值和导航语义是 privileged simulation measurements。不是从图像检测出来的状态。

## 动作

`float32[2]`，依次 steering ∈[−1,1]、signed pedal ∈[−1,1]。Native 归一化转向映射最大 0.45rad，pedal 正值最大 3m/s²，负值最大制动 −6m/s²，并有执行器滞后/阻力；MetaDrive 使用其原生方向盘、油门/制动语义，不硬凑第三维 brake。非有限或维度错误动作拒绝，越界值饱和。

## 策略与训练

`train(TrainConfig, output_dir, resume=None)` 按 algorithm 分派。`load_checkpoint(path)` 返回 policy、解析配置、载入元数据。`EndToEndAgent.predict(obs)` 只做神经策略推理；`act(obs_dict)` 共享策略处理多车。模型不调用规则控制器做隐藏修正。

MAPPO critic 输入维度是 `num_agents * (state_dim + 1) + num_agents`。最后 num_agents 项是 self ID。Actor 仍然只接收单车局部观测。

## 评估

`evaluate(agent_factory, env_config, seeds, output, label, save_trajectories)` 输出统一 dict，CSV/JSON/HTML/Markdown 共用该 dict。成功必须由环境显式到达终点信号触发；timeout 不算成功。评估配置若变更后端或观测结构，调用方必须保证模型 schema 兼容。

## 数据

Portable JSON schema_version=1，tracks 每行 `[time_s,x_m,y_m,yaw_rad,speed_mps]`；要求有限数、严格递增时间、非负速度、唯一 ID、来源声明。轨迹回放车按记录状态移动，不会响应 ego，因此与 IDM 背景模型的闭环交互性质不同。

## 配置组合与训练课程

`load_config()` 支持 `include: [env/native.yaml, algorithm/ppo.yaml]`；相对路径以包含文件为基准，后合并覆盖先合并，CLI overrides 最后覆盖。检测循环 include 与未知字段。每次训练写出完整解析配置，而非只保存引用。

`configs/curriculum.yaml` 将课程接入 PPO/IPPO/MAPPO 的**联合回合结束点**。20 个近期联合场景以上，联合成功率 ≥0.75 且任何车辆碰撞率 ≤0.15 才升级。阶段为 `(0车,直道)`、`(5车,幅度2米弯道)`、`(12车,幅度5米弯道)`。仅修改训练场景；checkpoint 的 config.env 保留目标评估配置，extra 保存课程阶段/历史。恢复时还原课程，不保证仿真轨迹逐位连续。不支持 SAC/MetaDrive 课程，显式报错。阶段日志在 `curriculum.csv`，未晋级也如实记录。

舒适度字段 `jerk_*_mps3` 仅度量纵向加速度的差分，不包括横向 jerk。高斯观测噪声仅用于状态向量有效字段，不是图像噪声/遮挡模型。
