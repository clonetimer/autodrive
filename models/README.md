# 实际模型索引

`registry.json` 的每项给出相对于项目根目录的真实 `.pt` 路径、SHA256、算法、输入和已执行世界步。权重仅保存一份，在 `experiments/release/<run>/model.pt` 和 `experiments/curriculum_release/model.pt`；无需联网下载。

这些不是预训练占位路径。权重通过实际 CPU 优化、保存重载测试。它们也不是可靠驾驶认证；尤其稠密交通与多车测试仍有高碰撞率，详见实验报告。

## v2.7 research candidate

`metadrive_ppo_state_expert_distilled_v27.pt` 由 v2.5 source policy 在 `expert_train` 域内接受 MetaDrive `IDMPolicy` training-only privileged lateral labels 后得到。它只在 validation2 通过工程 promotion gate；release6 没有确认稳健 raw-policy 安全收益，因此 registry 状态为 `validation2_promoted_release6_not_confirmed_not_safety_certified`。不得把该模型用于实车控制。

## v3.1 learned route adapter

`route_adapter_registry.json` 与标准 policy `registry.json` 分离。`metadrive_learned_route_adapter_v31.pt` 不是可独立执行的 DrivingPolicy checkpoint，而是绑定到 `metadrive_ppo_state_capability_distilled_v28.pt` SHA-256 的横向 residual adapter：输入 state + base action，输出 residual steering 与 learned blend，纵向动作保持不变。

训练使用 `adapter_train` 31000–31014 的真实 MetaDrive online DAgger，共 3,634 transitions。validation10 通过预声明 gate 后才冻结并打开 release14；release14 的 raw learned-adapter 系统为 60.0% success / 6.7% collision / 33.3% off-road，但 curve tracking 仍 0/3 success。运行时不调用 pure-pursuit RouteAssist；若再加 SafetySupervisor，则属于 learned-lateral + rule-longitudinal 混合系统。

该 artifact 的 SHA、base-model binding、训练数据指纹与 release 结果均在 `route_adapter_registry.json` 中。不要把 adapter 单独加入 `registry.json` 或宣称为安全认证模型。
