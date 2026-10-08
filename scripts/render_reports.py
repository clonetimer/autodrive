"""Render figures and write a Chinese report from recorded experiment JSON/CSV."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import _bootstrap
from src.utils.logger import save_json, sha256, runtime_manifest


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def render(root: Path):
    evaluation = read_json(root / 'reports/release_evaluation/summary.json')
    training = read_json(root / 'experiments/release/matrix.json')
    curriculum = read_json(root / 'experiments/curriculum_release/result.json')
    performance = read_json(root / 'reports/performance.json')
    images = root / 'reports/figures'
    images.mkdir(parents=True, exist_ok=True)
    for group, names in [('ppo_training', ['ppo_state_seed7', 'ppo_state_seed8', 'ppo_state_seed9']),
                         ('marl_training', ['mappo_state', 'mappo_hybrid', 'ippo_state'])]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for name in names:
            with (root / 'experiments/release' / name / 'training.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            ax.plot([int(r['world_steps']) for r in rows], [float(r['mean_recent_return']) for r in rows], label=name)
        ax.set_xlabel('Joint world steps')
        ax.set_ylabel('Recent 30 completed agent-episode returns')
        ax.set_title('Actual CPU training logs (not success-rate curves)')
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(images / f'{group}.png', dpi=150)
        plt.close(fig)
    groups = [('single_agent_success', evaluation['single_agent_test'], 'Success rate: 20 held-out single-ego scenarios'),
              ('multi_agent_joint_success', evaluation['multi_agent_test'], 'Joint success: 20 shared-world scenarios')]
    for filename, group, title in groups:
        names = list(group)
        values = [group[name]['metrics']['joint_success_rate'] for name in names]
        intervals = np.array([group[name]['metrics']['joint_success_wilson_ci95'] for name in names]).T
        errors = np.vstack((np.array(values)-intervals[0], intervals[1]-np.array(values)))
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(names, values, yerr=np.maximum(errors, 0), capsize=4)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('Success rate with 95% scenario Wilson interval')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        fig.savefig(images / f'{filename}.png', dpi=150)
        plt.close(fig)
    names = list(evaluation['generalization'])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, [evaluation['generalization'][name]['success_rate'] for name in names])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Success rate')
    ax.set_title('Frozen PPO seed 7: failures under distribution shift')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(images / 'generalization.png', dpi=150)
    plt.close(fig)

    totals = {'training_runs': len(training) + 1,
              'joint_world_steps': sum(r['world_steps_total'] for r in training.values()) + curriculum['world_steps_total'],
              'active_agent_transitions': sum(r.get('active_agent_steps_this_run', r['world_steps_total']) for r in training.values()) + curriculum['active_agent_steps_this_run'],
              'measured_training_seconds_sum': sum(r['elapsed_seconds'] for r in training.values()) + curriculum['elapsed_seconds'],
              'backend': 'native', 'real_metadrive_steps': 0,
              'disclaimer': 'No dev smoke/unit-test optimizer steps included. Not a convergence or safety claim.'}
    save_json(totals, root / 'reports/experiment_totals.json')
    save_json(runtime_manifest(), root / 'reports/runtime_manifest.json')
    registry_path = root / 'models/registry.json'
    try:
        registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else {}
    except (json.JSONDecodeError, OSError):
        registry = {}
    for name in training:
        path = root / 'experiments/release' / name / 'model.pt'
        registry[name] = {'path': str(path.relative_to(root)), 'sha256': sha256(path),
                          'backend': training[name]['backend'], 'input': training[name]['observation'],
                          'algorithm': training[name]['algorithm'], 'world_steps': training[name]['world_steps_total']}
    cp = root / 'experiments/curriculum_release/model.pt'
    registry['curriculum'] = {'path': str(cp.relative_to(root)), 'sha256': sha256(cp), 'backend': 'native',
                              'input': 'state', 'algorithm': 'ppo', 'world_steps': curriculum['world_steps_total']}
    save_json(registry, registry_path)

    lines = ['# 实际实验报告：v2.0.0 CPU research preview', '',
             '> 这是实际文件生成的结果，不是示例数字。MetaDrive 真正训练没有通过；下列训练、回放与评估全部为 native。', '',
             '## 协议与预算', '',
             f"共 **{totals['training_runs']} 次训练、{totals['joint_world_steps']:,} 个联合世界步、{totals['active_agent_transitions']:,} 个有效 Agent 转移**。",
             '一个世界步同时推进所有在场车辆，不得把世界步与单 Agent 样本相混。训练使用 CPU / 单 Torch 线程。',
             '单车 PPO 有 7/8/9 三个训练种子，其他配置单种子。测试场景固定为 2000–2019，验证诊断为 1000–1019；训练使用独立的训练根种子随机流。',
             '最终模型在测试前固定为预算终点权重，未依据测试结果回选最佳 checkpoint；低预算 checkpoint 只在验证集做诊断。',
             '主表各算法预算不同，MARL 的道路/车辆/随机化也不完全一致，因此不是严格的算法排名。95% Wilson 区间针对独立场景；同场景多个 Agent 不独立。', '',
             '## 训练产物', '', '| 运行 | 算法 / 输入 | 学习车辆 | 世界步 | 权重变化 L1 | CNN/状态 encoder 变化 L1 |',
             '|---|---|---:|---:|---:|---:|']
    for name, row in training.items():
        lines.append(f"| {name} | {row['algorithm']} / {row['observation']} | {row['num_agents']} | {row['world_steps_total']:,} | {row['parameters_changed_l1']:.2f} | {row['encoder_changed_l1']:.2f} |")
    lines += [f"| curriculum | ppo / state | 1 | {curriculum['world_steps_total']:,} | {curriculum['parameters_changed_l1']:.2f} | {curriculum['encoder_changed_l1']:.2f} |", '',
              '所有模型均有实际非零梯度/参数变化、权重文件、解析配置、训练 CSV、TensorBoard 标量文件和 SHA256。参数变化仅证明训练发生，不证明驾驶能力足够。', '',
              '## 单车测试', '',
              '| 策略 | 成功率 | 碰撞率 | 出界率 | 超时率 | 平均速度 m/s | 平均绝对纵向 jerk m/s³ | 推理 p95 ms |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, row in evaluation['single_agent_test'].items():
        m = row['metrics']
        lines.append(f"| {name} | {m['success_rate']:.1%} | {m['collision_rate']:.1%} | {m['out_of_road_rate']:.1%} | {m['timeout_rate']:.1%} | {m['avg_speed_mps']:.2f} | {m['jerk_mean_abs_mps3']:.2f} | {m['inference_p95_ms']:.3f} |")
    seed_rates = [evaluation['single_agent_test'][f'ppo_state_seed{seed}']['metrics']['success_rate'] for seed in [7, 8, 9]]
    lines += ['', f"三个 PPO 训练种子的成功率均值为 {np.mean(seed_rates):.1%}，样本标准差为 {np.std(seed_rates, ddof=1):.1%}；仅三个种子，不构成高精度估计。",
              '**规则基线在这组简单直道上优于 RL。** 不能把简单道路成功率解释为稠密交通驾驶能力。100%/0% 的有限样本结果也不是零风险；20/20 的 Wilson 下界约 83.9%，0/20 的碰撞率上界约 16.1%。', '',
              '## 多智能体测试', '',
              '| 策略 | 每车成功率 | 联合全部成功率 | 每车碰撞率 | 场景任一碰撞率 | 有效评估车辆回合 |',
              '|---|---:|---:|---:|---:|---:|']
    for name, row in evaluation['multi_agent_test'].items():
        m = row['metrics']
        lines.append(f"| {name} | {m['success_rate']:.1%} | {m['joint_success_rate']:.1%} | {m['collision_rate']:.1%} | {m['scenario_any_collision_rate']:.1%} | {m['agent_episodes']} |")
    lines += ['', 'MAPPO actor 仅使用本车局部观测，集中式 value 使用联合状态和 Agent 身份。shared-IPPO 使用局部 value。两者都是共同物理世界，并非独立环境拼接。',
              '**这些结果只证明 MARL 训练/执行链路已运行，碰撞水平明显未满足可靠驾驶要求。**', '',
              '## 固定策略泛化测试', '',
              '| 场景 | 成功率 | 碰撞率 | 出界率 |', '|---|---:|---:|---:|']
    for name, m in evaluation['generalization'].items():
        lines.append(f"| {name} | {m['success_rate']:.1%} | {m['collision_rate']:.1%} | {m['out_of_road_rate']:.1%} |")
    lines += ['', '**主要失败模式：高密度、慢前车和紧急制动时，短预算 PPO 常继续接近障碍物；弯道还出现横向出界。**',
              '进度奖励的权重和对安全距离缺少约束是待验证的解释，而不是已证明的单一原因。当前没有偷偷叠加规则安全盾来提高这些数字。', '',
              '## 验证集诊断与消融边界', '',
              '| 诊断 | 成功率 | 碰撞率 | 出界率 |', '|---|---:|---:|---:|']
    for name, row in evaluation['validation_diagnostics'].items():
        m = row['metrics']
        lines.append(f"| {name} | {m['success_rate']:.1%} | {m['collision_rate']:.1%} | {m['out_of_road_rate']:.1%} |")
    lines += ['', '奖励对比：同训练种子、预算与动力学，比较物理结果；不同奖励定义下的 return 不能横向比较。',
              'IPPO/MAPPO 对比：同种子、联合世界步与环境；终止行为不同使有效 Agent 样本数不同。',
              'State/BEV 对比：同种子/世界步/道路，但 rollout、batch size、更新轮次不同，因此只是输入配置诊断，不是隔离唯一变量的严格 encoder 消融。', '',
              '## 课程学习', '',
              f"已执行 {curriculum['world_steps_total']:,} 步，最终阶段为 **{curriculum['curriculum_final_stage']}**。门槛是近期至少20个联合回合、成功率≥75%、碰撞率≤15%。",
              '本次没有达到升级门槛，保留 stage=0，不伪造课程晋级。独立在目标三车测试环境评估的结果见 `reports/curriculum_evaluation`。', '',
              '## 性能实测', '',
              '| 观测 | 纯 step() 中位数 steps/s | 含 reset 的循环中位数 steps/s |', '|---|---:|---:|']
    for mode, row in performance['summary'].items():
        lines.append(f"| {mode} | {row['step_call_median_sps']:.1f} | {row['loop_median_sps']:.1f} |")
    lines += ['', '这里为2个学习车辆+请求8个IDM背景车辆，3次重复、每次500个世界步。step()包含动力学、碰撞检测、奖励和观测计算，但不含神经网络反向训练，也不是渲染FPS。',
              '状态输入超过500 steps/s；BEV/hybrid 在该配置下低于500，因此≥500的目标不能对所有输入模式宣称通过。所有 native 模式在该实测中超过100 steps/s。', '',
              '## 演示与可复验证据', '',
              '固定测试种子2000录制，不按结果挑选视频；每个GIF附逐步JSON，标明 native、输入模式、模型SHA256与终止状态。',
              '图表在 `reports/figures/`；逐回合JSON/CSV/HTML在 `reports/release_evaluation/`；模型索引在 `models/registry.json`。',
              'MetaDrive 的独立启动失败证据见 `reports/metadrive_doctor.json` 和 `experiments/metadrive_attempt/failure.json`。', '',
              '## 结论', '',
              '本轮实现并实测了 CPU 原生仿真、PPO/SAC、共享IPPO/CTDE MAPPO、语义BEV CNN、混合观测端到端控制、数据回放、自动评估与实验留痕。',
              '**不能将本交付写成“MetaDrive已完成训练”“原始相机端到端驾驶已完成”或“密集交通驾驶已达标”。** 最终状态是可运行研究版本、外部仿真验收阻塞，完整需求并未全部完成。', '']
    (root / 'docs/experiment_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return totals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(_bootstrap.ROOT))
    args = parser.parse_args()
    print(render(Path(args.root)))
