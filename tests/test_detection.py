import numpy as np
import pytest

from pulmoai.config import config_from_dict
from pulmoai.evaluation.metrics import bootstrap_cpm, detection_metrics
from pulmoai.inference.detector import candidates
from pulmoai.preprocessing.volume import Geometry


def _candidate(center, score=0.9):
    return {"center_xyz_mm": center, "score": score}


def _cohort():
    return [
        {
            "seriesuid": "a",
            "patient_id": "p1",
            "nodules": [{"center_xyz_mm": [0, 0, 0], "diameter_mm": 4}],
            "excluded": [{"center_xyz_mm": [10, 0, 0], "diameter_mm": 2}],
        },
        {"seriesuid": "b", "patient_id": "p2", "nodules": [], "excluded": []},
    ]


def test_exact_detection_counts_negative_scans_exclusions_duplicates():
    predictions = {
        "a": [_candidate([0, 0, 0]), _candidate([1, 0, 0], 0.8), _candidate([10, 0, 0], 0.7)],
        "b": [_candidate([0, 0, 0], 0.6)],
    }
    metrics = detection_metrics(_cohort(), predictions)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 2
    assert metrics["ignored_candidates"] == 1
    assert metrics["fp_per_scan"] == 1
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["cpm"] == 1


def test_tied_scores_cannot_artificially_reduce_false_positives():
    predictions = {"a": [_candidate([0, 0, 0])], "b": [_candidate([0, 0, 0])]}
    metrics = detection_metrics(_cohort(), predictions)
    assert metrics["sensitivity_at_fp_rates"][:2] == [0, 0]
    assert metrics["sensitivity_at_fp_rates"][2:] == [1] * 5
    assert metrics["cpm"] == pytest.approx(5 / 7)
    assert len(metrics["froc"]) == 2


def test_no_predictions_no_positives_and_unknown_series():
    assert detection_metrics(_cohort(), {})["sensitivity"] == 0
    assert detection_metrics([_cohort()[1]], {})["cpm"] is None
    with pytest.raises(ValueError, match="unknown"):
        detection_metrics(_cohort(), {"unknown": []})
    with pytest.raises(ValueError, match="Invalid"):
        detection_metrics(_cohort(), {"a": [_candidate([0, 0, 0], float("nan"))]})


def test_bootstrap_reproducible_patient_groups():
    predictions = {"a": [_candidate([0, 0, 0])]}
    first = bootstrap_cpm(_cohort(), predictions, [1], 100, 7)
    assert first == bootstrap_cpm(_cohort(), predictions, [1], 100, 7)
    assert first["interval_95"] == [1, 1]
    assert bootstrap_cpm([_cohort()[0]], predictions, [1], 10, 7)["interval_95"] is None


def test_candidate_nms_restores_original_coordinates_and_collapses_plateau():
    original = Geometry((16, 16, 16), (-30, 20, 4), (2, 2, 2), (0, -1, 0, 1, 0, 0, 0, 0, 1))
    processed = Geometry((32, 32, 32), original.origin_xyz, (1, 1, 1), original.direction)
    heatmap = np.zeros(processed.shape_zyx, dtype=np.float32)
    heatmap[8, 10, 12] = 0.9
    heatmap[8, 10, 14] = 0.8
    config = config_from_dict({"evaluation": {"score_threshold": 0.5}})
    found = candidates(heatmap, processed, original, config)
    assert len(found) == 1
    assert np.allclose(found[0]["center_voxel_xyz"], [6, 5, 4])
    assert np.allclose(found[0]["center_xyz_mm"], original.world([6, 5, 4]))
    heatmap.fill(0.7)
    assert len(candidates(heatmap, processed, original, config)) == 1


def test_included_annotations_take_precedence_over_overlapping_exclusions():
    cohort = _cohort()
    cohort[0]["excluded"] = cohort[0]["nodules"]
    found = {"a": [_candidate([0, 0, 0]), _candidate([1, 0, 0], 0.8)]}
    metrics = detection_metrics(cohort, found)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["ignored_candidates"] == 0
