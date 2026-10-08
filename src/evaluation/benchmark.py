"""Evaluate a frozen policy on disjoint procedural distribution shifts."""
from dataclasses import replace
from src.end_to_end.agent import EndToEndAgent
from src.evaluation.evaluator import evaluate


def generalization_suite(checkpoint, seeds, output):
    from pathlib import Path
    from src.utils.logger import save_json
    agent = EndToEndAgent(checkpoint)
    base = agent.config.env
    scenarios = {
        'matched': {},
        'dense': {'traffic_vehicles': 16},
        'curved': {'curve_amplitude': 6.0},
        'sensor_noise': {'observation_noise': 0.04},
        'slow_leader': {'scenario': 'follow'},
        'emergency': {'scenario': 'emergency'},
    }
    if base.backend != 'native':
        raise ValueError('These specific distribution shifts are native-road scenarios, not MetaDrive configuration aliases')
    reports = {}
    for label, changes in scenarios.items():
        report = evaluate(lambda name, seed: agent, replace(base, **changes), seeds,
                          Path(output) / label, label=label)
        reports[label] = report['metrics']
    save_json(reports, Path(output) / 'generalization.json')
    return reports
