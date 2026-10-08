# 已知限制与未通过目标

1. **MetaDrive 真正运行、训练和 BEV/MARL 集成尚未验证。** 上传源码不能替代 Panda3D 等二进制依赖与 assets。沙盒网络安装失败。真实环境测试显示 skipped，失败启动证据保留。该目标不能标为 completed。
2. **实际训练与演示都是 native 后端。** 原生模型是平面运动学自行车、简化道路与交通，没有高保真轮胎/碰撞响应、交通灯、复杂环岛或真实传感器噪声建模。原始要求中的所有地图类型未实现。
3. **BEV 是真值语义栅格。** 不是多相机感知，不包含 BEV 重建、遮挡推断、传感器标定或 learned world model。端到端范围是语义观测到控制。
4. **短预算训练不等于行为能力达标。** 必须读逐场景失败率；CNN 和多车策略可能比规则 baseline 更差。不能只引用单种简单道路成功率，不宣称≤2M步收敛或工业级安全。
5. **统计预算有限。** 三个训练种子只用于单车 PPO；其他算法/视觉配置不是全面多种子实验。不同模型预算、环境配置并不完全一致，主表是工程功能验证，不可作为严格算法优劣排名。
6. **MARL 是参数共享 feed-forward PPO / CTDE MAPPO。** 不包含 V2X、协同通信学习、完整原论文全部 tricks、RLlib/PettingZoo 官方兼容认证或无限动态 respawn。MetaDrive 场景的 MARL 验收仍阻塞。
7. **数据路径没有下载到真实 nuScenes/Waymo。** 提供通用授权下载、校验、CSV 转换、合成轨迹 fixture 和非响应式 replay。原始大数据集全量格式转换及真实 ScenarioNet 回放尚未测试。许可不能由脚本绕过。
8. **恢复范围明确。** PPO/MAPPO 是模型/优化器/计数器恢复，不是仿真世界 bitwise 恢复；SAC actor checkpoint 只支持推理，不支持完整断点续训。
9. **Gymnasium/TensorBoard/Docker 边界。** 本沙盒未运行真实 Gymnasium checker、TensorBoard Web UI 或 Docker build。TFRecord/CRC 和 protobuf schema 做了文件级验证，不能称为已运行 TensorBoard UI。
10. **舒适性指标仅为纵向加速度 jerk。** 未评估横向加速度/横向 jerk、转向率或乘坐舒适标准，不能以一个 jerk 均值代表完整舒适性。观测高斯噪声作用于状态特征，不代表真实相机噪声。
11. **性能实测不能跨平台担保。** 报告明确区分纯 env steps/s 与带推理的闭环 steps/s；不把渲染 FPS 等同于训练吞吐，也不承诺所有地图/后端 ≥500 FPS。

12. **全局环境不是本项目专用干净虚拟环境。** `pip check` 检测到预装 moviepy 对 Pillow<12 的约束冲突；本项目未使用 moviepy，GIF直接用Pillow生成并实测。具体版本和原始检查结果见 `reports/tested_environment.json`。不宣称整个沙盒所有软件依赖兼容。

没有 GPU 本身不是 MetaDrive 的阻塞理由。CPU 算法已实测；真正未满足的前提是运行依赖、assets 与下载网络。门禁通过前，发布状态为 CPU research preview，而不是全部目标通过的 stable release。
