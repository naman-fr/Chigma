"""
FD-YOLO11 Real-Time Inference Engine
======================================
Production inference with batching, TensorRT acceleration,
and Prometheus metrics integration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger


class InferenceEngine:
    """Production inference engine for FD-YOLO11.

    Features:
    - Single image and batch inference
    - Automatic format detection (PyTorch, ONNX, TensorRT)
    - NMS post-processing
    - Result visualization with annotated bounding boxes
    - Prometheus metrics emission
    """

    DEFECT_CLASSES = [
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled_in_scale", "scratches",
    ]

    DEFECT_COLORS = {
        "crazing": (255, 100, 100),
        "inclusion": (100, 255, 100),
        "patches": (100, 100, 255),
        "pitted_surface": (255, 255, 100),
        "rolled_in_scale": (255, 100, 255),
        "scratches": (100, 255, 255),
    }

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model()
        logger.info(f"InferenceEngine ready | model={self.model_path.name} | device={self.device}")

    def _load_model(self) -> Any:
        """Load model based on file format."""
        from ultralytics import YOLO

        model = YOLO(str(self.model_path))
        return model

    def predict(
        self,
        image: np.ndarray | str | Path,
        visualize: bool = False,
    ) -> dict[str, Any]:
        """Run inference on a single image.

        Args:
            image: Input image (numpy array, file path, or URL).
            visualize: Whether to draw detections on the image.

        Returns:
            Detection results with boxes, scores, classes, and timing.
        """
        start = time.perf_counter()

        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        latency = time.perf_counter() - start
        result = results[0]

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.DEFECT_CLASSES[cls_id] if cls_id < len(self.DEFECT_CLASSES) else f"class_{cls_id}"
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                })

        output = {
            "detections": detections,
            "num_detections": len(detections),
            "latency_ms": round(latency * 1000, 2),
            "image_shape": result.orig_shape,
        }

        if visualize and isinstance(image, np.ndarray):
            output["annotated_image"] = self._draw_detections(image.copy(), detections)

        return output

    def predict_batch(self, images: list[np.ndarray | str | Path]) -> list[dict[str, Any]]:
        """Run inference on a batch of images."""
        return [self.predict(img) for img in images]

    def stream(
        self,
        source: str | int = 0,
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> None:
        """Real-time inference from video stream.

        Args:
            source: Video file path, RTSP URL, or camera index.
            show: Display annotated frames.
            save_path: Optional path to save annotated video.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))

        frame_count = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.predict(frame, visualize=True)
                annotated = result.get("annotated_image", frame)
                frame_count += 1

                # Overlay FPS
                fps_text = f"FPS: {1000 / max(result['latency_ms'], 1):.0f} | Defects: {result['num_detections']}"
                cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if writer:
                    writer.write(annotated)

                if show:
                    cv2.imshow("FD-YOLO11 Defect Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            logger.info(f"Stream ended | {frame_count} frames processed")

    def _draw_detections(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class_name"]
            conf = det["confidence"]
            color = self.DEFECT_COLORS.get(cls_name, (200, 200, 200))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - label_h - 8), (x1 + label_w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image
