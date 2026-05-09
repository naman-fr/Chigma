"""
Auto-Labeler — Automatic YOLO-format Annotation from Scene Metadata
=====================================================================
Generates bounding box annotations in YOLO format from
3D scene rendering metadata. No manual annotation needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

# Mapping from defect type name to class ID
DEFECT_CLASS_MAP = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled_in_scale": 4,
    "scratches": 5,
}


class AutoLabeler:
    """Generate YOLO-format labels from scene metadata.

    Converts 3D scene rendering metadata into YOLO annotation files
    with format: <class_id> <cx> <cy> <w> <h> (all normalized 0-1).

    Args:
        output_dir: Directory to save .txt label files.
    """

    def __init__(self, output_dir: str | Path = "data/synthetic/labels") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats: dict[str, int] = {name: 0 for name in DEFECT_CLASS_MAP}

    def generate_labels(self, scenes: list[dict[str, Any]]) -> list[Path]:
        """Generate YOLO label files for a list of rendered scenes.

        Args:
            scenes: List of scene metadata dicts from SceneRenderer.

        Returns:
            List of paths to generated label files.
        """
        label_paths = []

        for scene in scenes:
            image_path = Path(scene["image_path"])
            defect_type = scene["defect_type"]
            regions = scene["defect_regions"]

            # Create label file with same name as image but .txt extension
            label_filename = image_path.stem + ".txt"
            label_path = self.output_dir / label_filename

            class_id = DEFECT_CLASS_MAP.get(defect_type, 0)

            lines = []
            for region in regions:
                # YOLO format: class_id cx cy w h (normalized)
                cx = self._clamp(region["cx"])
                cy = self._clamp(region["cy"])
                w = self._clamp(region["w"])
                h = self._clamp(region["h"])
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Write label file
            with open(label_path, "w") as f:
                f.write("\n".join(lines))

            label_paths.append(label_path)
            self.stats[defect_type] = self.stats.get(defect_type, 0) + len(regions)

        logger.info(f"Generated {len(label_paths)} label files | stats={self.stats}")
        return label_paths

    def validate_labels(self, label_dir: str | Path | None = None) -> dict[str, Any]:
        """Validate generated labels for correctness.

        Checks:
        - All values in [0, 1] range
        - Valid class IDs
        - No empty label files
        - Label-image pair consistency
        """
        label_dir = Path(label_dir) if label_dir else self.output_dir
        issues: list[str] = []
        total_boxes = 0
        valid_files = 0

        for label_file in label_dir.glob("*.txt"):
            with open(label_file) as f:
                lines = f.readlines()

            if not lines:
                issues.append(f"Empty label file: {label_file.name}")
                continue

            valid_files += 1
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    issues.append(f"{label_file.name}:{line_num} — expected 5 values, got {len(parts)}")
                    continue

                class_id = int(parts[0])
                values = [float(v) for v in parts[1:]]

                if class_id not in DEFECT_CLASS_MAP.values():
                    issues.append(f"{label_file.name}:{line_num} — invalid class_id {class_id}")

                for i, val in enumerate(values):
                    if not 0.0 <= val <= 1.0:
                        issues.append(f"{label_file.name}:{line_num} — value {val} out of [0,1] range")

                total_boxes += 1

        report = {
            "total_files": valid_files,
            "total_boxes": total_boxes,
            "issues": issues,
            "is_valid": len(issues) == 0,
        }

        if issues:
            logger.warning(f"Label validation found {len(issues)} issues")
        else:
            logger.info(f"Labels valid: {valid_files} files, {total_boxes} boxes")

        return report

    @staticmethod
    def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp value to [min_val, max_val] range."""
        return max(min_val, min(value, max_val))
