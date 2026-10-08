from pathlib import Path
import json
import pytest
from src.utils.segmented import segment_targets, completed_training_artifact


def test_segment_targets():
    assert segment_targets(0, 10000, 4096) == [4096, 8192, 10000]
    assert segment_targets(8192, 16384, 8192) == [16384]
    assert segment_targets(10, 10, 1) == []
    with pytest.raises(ValueError):
        segment_targets(11, 10, 1)


def test_completed_training_artifact(tmp_path: Path):
    (tmp_path / 'model.pt').write_bytes(b'model')
    (tmp_path / 'result.json').write_text(json.dumps({
        'status': 'training_completed_not_a_driving_safety_claim', 'world_steps_total': 20
    }))
    assert completed_training_artifact(tmp_path, 20)['world_steps_total'] == 20
    assert completed_training_artifact(tmp_path, 21) is None
