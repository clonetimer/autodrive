# 系统架构与设计变更

## 实际执行链路

```text
YAML + seed + backend
        |
        v
NativeWorld / MetaDriveParallelEnv / ReplayWorld
        | joint actions -> physics -> final observations
        v
Snapshot -> state builder / semantic BEV rasterizer
        |
        +--> local MLP / CNN / fusion --> Gaussian actor --> tanh controls
        |
        +--> local value (PPO/IPPO) OR joint state + agent ID (MAPPO critic)
        |
        v
rollout + valid-agent mask + terminated/truncated -> GAE -> PPO updates
        |                        OR
        +--> replay -> twin-Q SAC + entropy-temperature updates
        |
        v
checkpoint + CSV + TFRecord scalar events + runtime provenance
        |
        v
frozen-policy held-out evaluation -> metrics/CI -> reports -> GIF/trajectory
```

## 关键约束

本版从上传的 v1.0.0 代码真实审计开始，而不是把旧对话中的“完成”当成事实。原包包含双目录根、7 维 toy state、假 Gym fallback、没有训练记录、MetaDrive passthrough。审计记录 `INPUT_AUDIT.json`；原始上传压缩包未被修改。

为了在无网络、已有 CPU PyTorch 的沙盒实际训练，新增 native PyTorch 算法实现。SB3 不再是核心运行强依赖。这是明确的工程调整，不把自研 PPO 冒称为 SB3。`utils/spaces.py` 的无 Gymnasium 模式只是本地协议；真实 Gymnasium checker 单独跳过。

## 仿真层

NativeWorld 使用道路 y=A sin(2πx/λ) 与平移车道、带执行器滞后的运动学自行车模型、物理子步、定向包围盒 SAT 碰撞、IDM 跟驰交通。多个 RL 车辆共享同一 `vehicles` 字典和碰撞计算；不是多个独立单车环境的并列。每个车辆的动作先收集，再对子步共同世界积分。

Native 的曲线车道是对解析中心线的横向平移，不是严格等距 Frenet 道路；不模拟坡度、轮胎侧滑、真实交通灯或路口。MetaDrive 适配使用真正车辆、当前/下一导航车道与真实环境奖励，必须单独安装依赖；不隐式退回 native。

## 生命周期

`agents` 是当前仍活跃的学习车辆列表；`possible_agents` 在一次训练配置里固定。step 只接收所有活跃车辆的动作。返回数据包括刚结束车辆的最后观测，之后该车辆从 `agents` 删除。训练时失活车辆补零，损失由 valid mask 排除。`transition_state` 在移除车辆之前保存，供 time-limit bootstrap 使用。

MetaDrive 多智能体子类在 `_after_vehicle_done` 清理前触发快照回调，避免失效的车辆/导航对象导致终态损坏。此 hook 根据用户提供的 0.4.3 源码设计，真实集成验收仍未通过。

## 产物与错误行为

配置外置且保存完整解析结果；训练目录默认不得覆盖。模型原子保存，并含算法、观测 schema、网络结构、全局步数。加载用 `torch.load(..., weights_only=True)`。下载采用 `.part`、大小上限、SHA-256；ZIP 检查越界路径及符号链接。

报错不是成功产物：MetaDrive 依赖不足写入 `failure.json` 后退出；没有 weights，也没有假训练曲线。完整版本验收由 `RELEASE_STATUS.json` 表达，不由文件名决定。


## v2.5 Learned-policy / safety-system feedback loop

```text
state observation -> PPO actor -> proposed action
                           |
                           v
                  SafetySupervisor ----> applied action -> MetaDrive
                           |
                           +---- logged correction dataset
                                      |
                                      v
                           weighted actor distillation
                                      |
                                      v
                            distilled raw actor
```

训练和报告层始终分开 `raw actor` 与 `actor + supervisor`。监督系统的成功率不能回填为raw actor指标；release报告同时保留介入时间、修正幅度和失败轨迹。
