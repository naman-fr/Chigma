"""
Drone Perception — YOLO11 Real-time Object Detection for Aerial Imagery
=========================================================================
Lightweight perception pipeline optimized for onboard drone processing.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from loguru import logger


class DronePerception:
    """Real-time perception for drone autonomy.

    Uses YOLO11n (lightweight) for real-time object detection
    optimized for aerial perspectives and edge deployment.

    Features:
    - Multi-object tracking (BoT-SORT/ByteTrack)
    - Obstacle classification and distance estimation
    - Frame-rate monitoring for safety constraints
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        tracker: str = "botsort.yaml",
        conf: float = 0.35,
        iou: float = 0.45,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.tracker = tracker
        self.conf = conf
        self.iou = iou
        self.device = device
        self._model = None
        self._frame_count = 0
        self._fps_history: list[float] = []

    @property
    def model(self):
        """Lazy-load YOLO model."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
            logger.info(f"Perception model loaded: {self.model_name}")
        return self._model

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        """Run detection on a single frame.

        Args:
            frame: BGR image from drone camera.

        Returns:
            Detection results with tracked objects.
        """
        start = time.perf_counter()

        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )

        latency = time.perf_counter() - start
        fps = 1.0 / max(latency, 1e-6)
        self._fps_history.append(fps)
        self._frame_count += 1

        result = results[0]
        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                det = {
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [float(x) for x in box.xyxy[0].tolist()],
                    "track_id": int(box.id[0]) if box.id is not None else None,
                }
                detections.append(det)

        return {
            "detections": detections,
            "frame_id": self._frame_count,
            "latency_ms": round(latency * 1000, 2),
            "fps": round(fps, 1),
            "avg_fps": round(np.mean(self._fps_history[-30:]), 1),
        }

    def classify_obstacles(self, detections: list[dict]) -> list[dict]:
        """Classify detected objects as obstacles or targets."""
        obstacle_classes = {"person", "car", "truck", "bus", "bird", "tree", "building"}

        for det in detections:
            det["is_obstacle"] = det["class_name"].lower() in obstacle_classes
            # Rough distance estimation from bbox size
            bbox = det["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            det["estimated_distance_m"] = max(1.0, 500000.0 / max(area, 1.0))

        return detections
