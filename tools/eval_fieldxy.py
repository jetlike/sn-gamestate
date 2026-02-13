#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _finite_xy(xy: Any) -> bool:
    if not isinstance(xy, (list, tuple)) or len(xy) != 2:
        return False
    try:
        x = float(xy[0])
        y = float(xy[1])
    except Exception:
        return False
    return math.isfinite(x) and math.isfinite(y)


def _greedy_match(pred: list[tuple[float, float]], gt: list[tuple[float, float]]) -> list[float]:
    if not pred or not gt:
        return []

    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            pairs.append((_dist(p, g), i, j))
    pairs.sort(key=lambda x: x[0])

    used_p: set[int] = set()
    used_g: set[int] = set()
    dists: list[float] = []
    for d, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        dists.append(d)
    return dists


def _percentile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    idx = int((len(vals) - 1) * q)
    return vals[idx]


def load_predictions(jsonl_path: Path) -> dict[int, dict[str, list[tuple[float, float]]]]:
    by_frame: dict[int, dict[str, list[tuple[float, float]]]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            frame = int(row.get("frame", -1))
            if frame < 0:
                continue
            objs = row.get("objects", [])
            players: list[tuple[float, float]] = []
            balls: list[tuple[float, float]] = []
            for o in objs:
                typ = o.get("type")
                xy = o.get("field_xy")
                if not _finite_xy(xy):
                    continue
                pt = (float(xy[0]), float(xy[1]))
                if typ in {"player", "goalkeeper"}:
                    players.append(pt)
                elif typ == "ball":
                    balls.append(pt)
            by_frame[frame] = {"player": players, "ball": balls}
    return by_frame


def load_ground_truth(labels_path: Path) -> dict[int, dict[str, list[tuple[float, float]]]]:
    data = json.loads(labels_path.read_text(encoding="utf-8"))

    image_id_to_frame: dict[str, int] = {}
    for im in data.get("images", []):
        # file_name like 000001.jpg -> frame index 0
        fname = str(im.get("file_name", ""))
        if fname.endswith(".jpg") and len(fname) >= 5:
            frame = int(Path(fname).stem) - 1
            image_id_to_frame[str(im["image_id"])] = frame

    by_frame: dict[int, dict[str, list[tuple[float, float]]]] = {}
    for ann in data.get("annotations", []):
        if ann.get("supercategory") != "object":
            continue
        attrs = ann.get("attributes", {}) or {}
        role = attrs.get("role")
        bbox_pitch = ann.get("bbox_pitch")
        if not isinstance(bbox_pitch, dict):
            continue
        x = bbox_pitch.get("x_bottom_middle")
        y = bbox_pitch.get("y_bottom_middle")
        if x is None or y is None:
            continue
        try:
            pt = (float(x), float(y))
        except Exception:
            continue
        image_id = str(ann.get("image_id"))
        frame = image_id_to_frame.get(image_id)
        if frame is None:
            continue
        if frame not in by_frame:
            by_frame[frame] = {"player": [], "ball": []}
        if role in {"player", "goalkeeper"}:
            by_frame[frame]["player"].append(pt)
        elif role == "ball":
            by_frame[frame]["ball"].append(pt)
    return by_frame


def evaluate(
    pred: dict[int, dict[str, list[tuple[float, float]]]],
    gt: dict[int, dict[str, list[tuple[float, float]]]],
    player_thr_m: float,
    ball_thr_m: float,
) -> dict[str, Any]:
    frames = sorted(set(gt.keys()) | set(pred.keys()))

    p_dists: list[float] = []
    b_dists: list[float] = []

    p_tp = p_fp = p_fn = 0
    b_tp = b_fp = b_fn = 0

    frames_with_gt_ball = 0
    frames_with_pred_ball = 0
    frames_with_ball_match = 0

    for fr in frames:
        pp = pred.get(fr, {}).get("player", [])
        pb = pred.get(fr, {}).get("ball", [])
        gp = gt.get(fr, {}).get("player", [])
        gb = gt.get(fr, {}).get("ball", [])

        # Players (greedy one-to-one)
        pd = _greedy_match(pp, gp)
        p_dists.extend(pd)
        p_frame_tp = sum(1 for d in pd if d <= player_thr_m)
        p_tp += p_frame_tp
        p_fp += max(0, len(pp) - p_frame_tp)
        p_fn += max(0, len(gp) - p_frame_tp)

        # Ball (single best match semantics)
        if gb:
            frames_with_gt_ball += 1
        if pb:
            frames_with_pred_ball += 1
        bd = _greedy_match(pb, gb)
        if bd:
            dmin = min(bd)
            b_dists.append(dmin)
            if dmin <= ball_thr_m:
                frames_with_ball_match += 1
                b_tp += 1
            else:
                b_fn += 1
        elif gb:
            b_fn += 1

        if pb:
            # count one primary predicted ball per frame as positive for precision-like score
            if not gb:
                b_fp += 1
            elif not bd or min(bd) > ball_thr_m:
                b_fp += 1

    def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    p_prec, p_rec, p_f1 = prf(p_tp, p_fp, p_fn)
    b_prec, b_rec, b_f1 = prf(b_tp, b_fp, b_fn)

    return {
        "n_frames_eval": len(frames),
        "player": {
            "threshold_m": player_thr_m,
            "tp": p_tp,
            "fp": p_fp,
            "fn": p_fn,
            "precision": p_prec,
            "recall": p_rec,
            "f1": p_f1,
            "matched_count": len(p_dists),
            "median_error_m": _percentile(p_dists, 0.5),
            "p90_error_m": _percentile(p_dists, 0.9),
        },
        "ball": {
            "threshold_m": ball_thr_m,
            "tp": b_tp,
            "fp": b_fp,
            "fn": b_fn,
            "precision": b_prec,
            "recall": b_rec,
            "f1": b_f1,
            "matched_count": len(b_dists),
            "median_error_m": _percentile(b_dists, 0.5),
            "p90_error_m": _percentile(b_dists, 0.9),
            "frames_with_gt_ball": frames_with_gt_ball,
            "frames_with_pred_ball": frames_with_pred_ball,
            "frames_with_ball_match": frames_with_ball_match,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate field-XY JSONL against SoccerNet GT")
    ap.add_argument("--pred", required=True, help="Path to positions.jsonl")
    ap.add_argument("--labels", required=True, help="Path to Labels-GameState.json")
    ap.add_argument("--player-thr", type=float, default=5.0, help="Player match threshold in meters")
    ap.add_argument("--ball-thr", type=float, default=3.0, help="Ball match threshold in meters")
    ap.add_argument("--out", default=None, help="Optional JSON report output path")
    args = ap.parse_args()

    pred = load_predictions(Path(args.pred))
    gt = load_ground_truth(Path(args.labels))
    report = evaluate(pred, gt, args.player_thr, args.ball_thr)

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
