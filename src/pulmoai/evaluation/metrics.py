"""Score-tied, one-to-one physical matching with a conservative internal FROC."""

from __future__ import annotations

import numpy as np


def _events(scans: list[dict], predictions: dict[str, list[dict]]):
    events = []
    for scan in scans:
        matched = set()
        for prediction in sorted(
            predictions.get(scan["seriesuid"], []), key=lambda p: (-p["score"], *p["center_xyz_mm"])
        ):
            center = np.asarray(prediction["center_xyz_mm"])
            hits = [
                (np.linalg.norm(center - n["center_xyz_mm"]), i)
                for i, n in enumerate(scan["nodules"])
                if np.linalg.norm(center - n["center_xyz_mm"]) <= n["diameter_mm"] / 2
            ]
            eligible = [(distance, i) for distance, i in hits if i not in matched]
            if eligible:
                _, index = min(eligible)
                matched.add(index)
                kind = 0
            elif hits:
                kind = 1
            elif any(
                np.linalg.norm(center - n["center_xyz_mm"]) <= n["diameter_mm"] / 2
                for n in scan.get("excluded", [])
            ):
                kind = 2
            else:
                kind = 1
            events.append((prediction["score"], kind))
    return events


def detection_metrics(
    scans: list[dict],
    predictions: dict[str, list[dict]],
    fp_rates=(0.125, 0.25, 0.5, 1, 2, 4, 8),
    operating_threshold: float = 0.5,
) -> dict:
    if not scans:
        raise ValueError("Detection evaluation requires at least one scan")
    known = {s["seriesuid"] for s in scans}
    if len(known) != len(scans) or set(predictions) - known:
        raise ValueError("Predictions contain unknown series or duplicate scan records")
    for found in predictions.values():
        for p in found:
            if (
                len(p["center_xyz_mm"]) != 3
                or not np.isfinite(p["center_xyz_mm"]).all()
                or not np.isfinite(p["score"])
                or not 0 <= p["score"] <= 1
            ):
                raise ValueError("Invalid candidate score or coordinates")
    positives = sum(len(s["nodules"]) for s in scans)
    events = _events(scans, predictions)
    grouped = {}
    for score, kind in events:
        grouped.setdefault(score, [0, 0, 0])[kind] += 1
    curve = [{"threshold": None, "fp_per_scan": 0.0, "sensitivity": 0.0 if positives else None}]
    tp = fp = 0
    for threshold in sorted(grouped, reverse=True):
        tp += grouped[threshold][0]
        fp += grouped[threshold][1]
        curve.append(
            {
                "threshold": threshold,
                "fp_per_scan": fp / len(scans),
                "sensitivity": tp / positives if positives else None,
            }
        )
    sensitivities = (
        [max(p["sensitivity"] for p in curve if p["fp_per_scan"] <= rate) for rate in fp_rates]
        if positives
        else [None] * len(fp_rates)
    )
    tp, fp, ignored = [
        sum(counts[k] for score, counts in grouped.items() if score >= operating_threshold)
        for k in range(3)
    ]
    fn = positives - tp
    return {
        "protocol": "pulmoai_internal_froc_v1",
        "scan_count": len(scans),
        "nodule_count": positives,
        "operating_threshold": operating_threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "ignored_candidates": ignored,
        "sensitivity": tp / positives if positives else None,
        "fp_per_scan": fp / len(scans),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / positives if positives else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "fp_rates": list(fp_rates),
        "sensitivity_at_fp_rates": sensitivities,
        "cpm": float(np.mean(sensitivities)) if positives else None,
        "froc": curve,
        "not_estimable_reason": None if positives else "No annotated nodules in this cohort",
        "matching": "score-ordered nearest unmatched nodule within diameter/2; duplicates are FP",
        "interpolation": "maximum observed sensitivity within FP budget; no interpolation",
        "official_luna16_score": False,
    }


def bootstrap_cpm(
    scans: list[dict], predictions: dict[str, list[dict]], fp_rates, samples: int, seed: int
) -> dict:
    patients = sorted({s["patient_id"] for s in scans})
    if samples < 1 or len(patients) < 2:
        return {
            "interval_95": None,
            "reason": "Requires >=2 patients and bootstrap_samples >0",
            "valid_replicates": 0,
        }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        cohort, scores = [], {}
        for draw, patient in enumerate(rng.choice(patients, len(patients), replace=True)):
            for scan in scans:
                if scan["patient_id"] == patient:
                    uid = f"{draw}:{scan['seriesuid']}"
                    cohort.append({**scan, "seriesuid": uid})
                    scores[uid] = predictions.get(scan["seriesuid"], [])
        result = detection_metrics(cohort, scores, fp_rates)["cpm"]
        if result is not None:
            values.append(result)
    interval = np.percentile(values, [2.5, 97.5]).tolist() if len(values) >= 2 else None
    return {
        "interval_95": interval,
        "method": "patient-cluster percentile bootstrap",
        "requested_replicates": samples,
        "valid_replicates": len(values),
        "seed": seed,
        "reason": None if interval else "Insufficient positive replicates",
    }
