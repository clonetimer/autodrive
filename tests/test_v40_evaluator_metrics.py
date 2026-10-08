"""v4 evaluator must retain joint-adapter diagnostics when merging stages."""
from src.evaluation.learned_adapter_metrics import EpisodeMetrics, summarize


def test_joint_adapter_longitudinal_correction_is_aggregated():
    ep = EpisodeMetrics(seed=1, agent='agent0')
    info = {
        'dt': .1, 'speed': 5., 'lane_offset': 0., 'lane_width': 3.5,
        'jerk': 0., 'progress_m': .5, 'collision': False, 'out_of_road': False,
        'success': False, 'route_completion': .2, 'lane_changes': 0,
        'proposed_action': [0., .5], 'applied_action': [0., .5],
        'route_assist_blend': 0., 'route_assist_need_score': 0.,
        'learned_adapter_blend': .1,
        'learned_adapter_steering_correction': .02,
        'learned_adapter_longitudinal_correction': .3,
        'learned_adapter_neighbor_attention_mass': .5,
        'learned_adapter_visible_neighbors': 2.,
        'learned_adapter_history_frames': 6.,
        'risk': {'ttc_s': None},
    }
    ep.update(0., info, False, False, 1.)
    row = ep.result()
    assert row['learned_adapter_mean_longitudinal_correction'] == .3
    assert row['learned_adapter_peak_longitudinal_correction'] == .3
    merged = summarize([row], [1.])
    assert merged['learned_adapter_mean_longitudinal_correction'] == .3
    assert merged['learned_adapter_peak_longitudinal_correction'] == .3
