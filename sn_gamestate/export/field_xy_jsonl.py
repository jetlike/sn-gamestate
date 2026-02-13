import json
import logging
from pathlib import Path

import pandas as pd

from tracklab.pipeline import VideoLevelModule

log = logging.getLogger(__name__)


class FieldXYJsonlExporter(VideoLevelModule):
    # Note: video_id, frame, file_path are accessed from image metadatas at
    # runtime but are NOT declared in input_columns because they come from the
    # dataset, not from upstream pipeline modules.  The pipeline validator only
    # tracks columns produced by modules, so listing them here causes a
    # validation error when no state file is loaded.
    input_columns = {
        "detection": ["bbox_pitch", "bbox_conf", "role", "category_name", "bbox_ltwh"],
        "image": [],
    }
    output_columns = {
        "detection": [],
        "image": [],
    }

    def __init__(
        self,
        output_path: str,
        fps: float = 25.0,
        infer_goalkeeper_candidates: bool = True,
        include_image_bbox: bool = True,
        min_player_conf: float = 0.4,
        min_ball_conf: float = 0.6,
        emit_single_ball: bool = True,
        ball_track_max_step_m: float = 12.0,
        pitch_x_min: float = -52.5,
        pitch_x_max: float = 52.5,
        pitch_y_min: float = -34.0,
        pitch_y_max: float = 34.0,
        field_margin_m: float = 5.0,
        **kwargs,
    ):
        super().__init__()
        self.output_path = Path(output_path)
        self.fps = fps
        self.infer_goalkeeper_candidates = infer_goalkeeper_candidates
        self.include_image_bbox = include_image_bbox
        self.min_player_conf = min_player_conf
        self.min_ball_conf = min_ball_conf
        self.emit_single_ball = emit_single_ball
        self.ball_track_max_step_m = ball_track_max_step_m
        self.pitch_x_min = pitch_x_min
        self.pitch_x_max = pitch_x_max
        self.pitch_y_min = pitch_y_min
        self.pitch_y_max = pitch_y_max
        self.field_margin_m = field_margin_m

    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        image_rows = metadatas.sort_values(by="frame")
        grouped = detections.groupby("image_id") if len(detections) else {}
        prev_ball_xy = None

        with self.output_path.open("w", encoding="utf-8") as f:
            for image_id, img_meta in image_rows.iterrows():
                frame_dets = (
                    grouped.get_group(image_id).copy()
                    if image_id in grouped.groups
                    else pd.DataFrame(columns=detections.columns)
                )

                frame_objects = []
                ball_xy = None
                for det_id, row in frame_dets.iterrows():
                    bbox_pitch = row.get("bbox_pitch")
                    if not isinstance(bbox_pitch, dict):
                        continue

                    x = bbox_pitch.get("x_bottom_middle")
                    y = bbox_pitch.get("y_bottom_middle")
                    if x is None or y is None:
                        continue

                    role = row.get("role")
                    if not isinstance(role, str) or not role:
                        role = "player"

                    conf = float(row.get("bbox_conf", 0.0))
                    if role == "ball" and conf < self.min_ball_conf:
                        continue
                    if role in {"player", "goalkeeper"} and conf < self.min_player_conf:
                        continue

                    x = float(x)
                    y = float(y)
                    if not self._in_pitch_bounds(x, y):
                        continue

                    obj = {
                        "detection_id": int(det_id),
                        "type": role,
                        "field_xy": [x, y],
                        "confidence": conf,
                        "category_name": str(row.get("category_name", "")),
                    }

                    if self.include_image_bbox:
                        ltwh = row.get("bbox_ltwh")
                        if ltwh is not None:
                            obj["bbox_ltwh"] = [float(v) for v in ltwh]

                    frame_objects.append(obj)

                frame_objects, prev_ball_xy = self._select_ball(frame_objects, prev_ball_xy)
                ball_xy = next((o["field_xy"] for o in frame_objects if o["type"] == "ball"), None)

                if self.infer_goalkeeper_candidates:
                    self._mark_goalkeeper_candidates(frame_objects)

                frame_idx = int(img_meta.get("frame", -1))
                payload = {
                    "image_id": int(image_id),
                    "video_id": int(img_meta.get("video_id", -1)),
                    "frame": frame_idx,
                    "timestamp_sec": round(frame_idx / self.fps, 4) if frame_idx >= 0 else None,
                    "file_path": str(img_meta.get("file_path", "")),
                    "n_players": sum(1 for o in frame_objects if o["type"] == "player"),
                    "n_balls": sum(1 for o in frame_objects if o["type"] == "ball"),
                    "ball_xy": ball_xy,
                    "objects": frame_objects,
                }
                f.write(json.dumps(payload) + "\n")

        log.info("Wrote %d frames to %s", len(image_rows), self.output_path)
        return detections

    def _in_pitch_bounds(self, x: float, y: float) -> bool:
        return (
            self.pitch_x_min - self.field_margin_m <= x <= self.pitch_x_max + self.field_margin_m
            and self.pitch_y_min - self.field_margin_m <= y <= self.pitch_y_max + self.field_margin_m
        )

    def _select_ball(self, frame_objects, prev_ball_xy):
        balls = [o for o in frame_objects if o["type"] == "ball"]
        others = [o for o in frame_objects if o["type"] != "ball"]
        if not balls:
            return frame_objects, prev_ball_xy
        if not self.emit_single_ball:
            best = max(balls, key=lambda o: o["confidence"])
            return others + balls, best["field_xy"]

        if prev_ball_xy is None:
            chosen = max(balls, key=lambda o: o["confidence"])
            return others + [chosen], chosen["field_xy"]

        def score(ball_obj):
            bx, by = ball_obj["field_xy"]
            dx = bx - prev_ball_xy[0]
            dy = by - prev_ball_xy[1]
            d = (dx * dx + dy * dy) ** 0.5
            if d > self.ball_track_max_step_m:
                return ball_obj["confidence"] - 2.0
            return ball_obj["confidence"] - 0.03 * d

        chosen = max(balls, key=score)
        bx, by = chosen["field_xy"]
        d = ((bx - prev_ball_xy[0]) ** 2 + (by - prev_ball_xy[1]) ** 2) ** 0.5
        if d > self.ball_track_max_step_m and chosen["confidence"] < 0.85:
            return others, prev_ball_xy
        return others + [chosen], chosen["field_xy"]

    @staticmethod
    def _mark_goalkeeper_candidates(frame_objects):
        """Heuristic goalkeeper tagging using pitch-zone proximity to goal lines.

        Standard pitch: 105 x 68 m, origin at center.
        Goal lines at x ~ -52.5 (left) and x ~ +52.5 (right).
        Penalty area extends 16.5 m from the goal line.
        We tag the player closest to each goal line as a GK candidate,
        but only if they are inside the penalty-area x-range.
        """
        # Pitch half-length; penalty area depth from goal line.
        HALF_LENGTH = 52.5
        PENALTY_DEPTH = 16.5

        players = [obj for obj in frame_objects if obj.get("type") == "player"]
        if len(players) < 2:
            return

        # Players in the left penalty zone (x < -HALF_LENGTH + PENALTY_DEPTH)
        left_zone = [p for p in players if p["field_xy"][0] < -HALF_LENGTH + PENALTY_DEPTH]
        # Players in the right penalty zone (x > HALF_LENGTH - PENALTY_DEPTH)
        right_zone = [p for p in players if p["field_xy"][0] > HALF_LENGTH - PENALTY_DEPTH]

        if left_zone:
            gk_left = min(left_zone, key=lambda o: o["field_xy"][0])
            gk_left["goalkeeper_candidate"] = True
            gk_left["goalkeeper_side"] = "left"

        if right_zone:
            gk_right = max(right_zone, key=lambda o: o["field_xy"][0])
            gk_right["goalkeeper_candidate"] = True
            gk_right["goalkeeper_side"] = "right"

        # Fallback: if no player was in either penalty zone, pick the extreme players.
        if not left_zone and not right_zone:
            left = min(players, key=lambda o: o["field_xy"][0])
            right = max(players, key=lambda o: o["field_xy"][0])
            left["goalkeeper_candidate"] = True
            left["goalkeeper_side"] = "left"
            right["goalkeeper_candidate"] = True
            right["goalkeeper_side"] = "right"
