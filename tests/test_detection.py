"""Tests for FD-YOLO11 detection module."""

import torch
import pytest


class TestSCC3k2:
    """Test Self-Calibrated C3k2 module."""

    def test_output_shape(self):
        from src.detection.sc_c3k2 import SCC3k2
        module = SCC3k2(in_channels=256, out_channels=256, n_blocks=2)
        x = torch.randn(2, 256, 32, 32)
        out = module(x)
        assert out.shape == (2, 256, 32, 32)

    def test_different_channels(self):
        from src.detection.sc_c3k2 import SCC3k2
        module = SCC3k2(in_channels=128, out_channels=256, n_blocks=3)
        x = torch.randn(1, 128, 16, 16)
        out = module(x)
        assert out.shape == (1, 256, 16, 16)


class TestFSPPF:
    """Test Feature-fusion SPPF module."""

    def test_output_shape(self):
        from src.detection.fsppf import FSPPF
        module = FSPPF(in_channels=512, out_channels=512)
        x = torch.randn(2, 512, 20, 20)
        out = module(x)
        assert out.shape == (2, 512, 20, 20)

    def test_residual_dimension_match(self):
        from src.detection.fsppf import FSPPF
        module = FSPPF(in_channels=256, out_channels=512)
        x = torch.randn(1, 256, 16, 16)
        out = module(x)
        assert out.shape == (1, 512, 16, 16)


class TestDySample:
    """Test DySample upsampling module."""

    def test_upsample_2x(self):
        from src.detection.dysample import DySample
        module = DySample(in_channels=256, scale_factor=2)
        x = torch.randn(2, 256, 16, 16)
        out = module(x)
        assert out.shape == (2, 256, 32, 32)


class TestFDYOLO11:
    """Test complete FD-YOLO11 model."""

    def test_forward_pass(self):
        from src.detection.fd_yolo11 import FDYOLO11
        model = FDYOLO11(num_classes=6, base_channels=32)
        x = torch.randn(1, 3, 320, 320)
        outputs = model(x)
        assert len(outputs) == 3  # 3 scale levels
        for cls_pred, reg_pred in outputs:
            assert cls_pred.shape[1] == 6  # num_classes
            assert reg_pred.shape[1] == 64  # 4 * reg_max

    def test_parameter_count(self):
        from src.detection.fd_yolo11 import FDYOLO11
        model = FDYOLO11(num_classes=6, base_channels=32)
        params = model.count_parameters()
        assert params > 0
        assert params < 100_000_000  # Less than 100M


class TestAutoLabeler:
    """Test automatic label generation."""

    def test_label_generation(self, tmp_path):
        from src.syngen.auto_label import AutoLabeler
        labeler = AutoLabeler(output_dir=tmp_path)

        scenes = [
            {
                "image_path": str(tmp_path / "test_001.png"),
                "defect_type": "scratches",
                "defect_regions": [
                    {"cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.2},
                ],
            }
        ]

        paths = labeler.generate_labels(scenes)
        assert len(paths) == 1

        with open(paths[0]) as f:
            line = f.readline().strip()
            parts = line.split()
            assert len(parts) == 5
            assert parts[0] == "5"  # scratches class_id

    def test_label_validation(self, tmp_path):
        from src.syngen.auto_label import AutoLabeler
        labeler = AutoLabeler(output_dir=tmp_path)

        # Write a valid label
        label_file = tmp_path / "test.txt"
        label_file.write_text("0 0.5 0.5 0.3 0.2\n")

        report = labeler.validate_labels(tmp_path)
        assert report["is_valid"]
        assert report["total_boxes"] == 1


class TestVLMCommandParser:
    """Test natural language command parsing."""

    def test_fly_to(self):
        from src.drone.vlm_commands import VLMCommandParser
        parser = VLMCommandParser()
        result = parser.parse("fly to the red building")
        assert result["action"] == "fly_to"
        assert result["target"] == "red building"

    def test_return_home(self):
        from src.drone.vlm_commands import VLMCommandParser
        parser = VLMCommandParser()
        result = parser.parse("return home")
        assert result["action"] == "return_home"

    def test_with_altitude(self):
        from src.drone.vlm_commands import VLMCommandParser
        parser = VLMCommandParser()
        result = parser.parse("fly to the tower at 50 meters")
        assert result["action"] == "fly_to"
        assert result.get("altitude_m") == 50.0

    def test_unknown_command(self):
        from src.drone.vlm_commands import VLMCommandParser
        parser = VLMCommandParser()
        result = parser.parse("xyzzy blah blah")
        assert result["action"] == "unknown"
