# 主要来源与对照范围

访问日期：2026-09-07。实现不是这些项目的官方复现，未复制整段实现。以下仅用作算法、接口和公开文件格式依据。

1. Schulman et al., Proximal Policy Optimization Algorithms, 2017. https://arxiv.org/abs/1707.06347
2. Gymnasium, Handling Time Limits. https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
3. Yu et al., The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games. https://arxiv.org/abs/2103.01955
4. Haarnoja et al., Soft Actor-Critic Algorithms and Applications. https://arxiv.org/abs/1812.05905
5. MetaDrive, Environments / Action and Policy / Training. https://metadrive-simulator.readthedocs.io/en/latest/rl_environments.html ; https://metadrive-simulator.readthedocs.io/en/latest/action.html ; https://metadrive-simulator.readthedocs.io/en/latest/training.html
6. MetaDrive, Installing MetaDrive / Scenario Description. https://metadrive-simulator.readthedocs.io/en/latest/install.html ; https://metadrive-simulator.readthedocs.io/en/latest/scenario_description.html
7. TensorBoard Event protobuf schema. https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/event.proto
8. TensorBoard Summary protobuf schema. https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/summary.proto
9. TensorBoard TFRecord writer. https://github.com/tensorflow/tensorboard/blob/master/tensorboard/summary/writer/record_writer.py
10. SB3 VecEnv terminal-observation convention（仅参考，不是本版实际执行框架）. https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html

MetaDrive 的具体属性还逐项对照了用户上传的 0.4.3 源码：`version.py`、`base_class/base_object.py`（speed vs speed_km_h）、`component/lane/abs_lane.py`、`envs/base_env.py`、`envs/scenario_env.py`、`envs/marl_envs/multi_agent_metadrive.py`、`manager/agent_manager.py`。源码对照不等于 runtime 集成通过。

依赖与上传源码未重新分发进最终仓库压缩包，各自遵循原项目许可。
