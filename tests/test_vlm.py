"""Tests for VLM Copilot module."""
import pytest

pytest.importorskip("torch")

from unittest.mock import MagicMock, patch


class TestVLMCopilot:
    """Test VLM Copilot integration."""

    @pytest.mark.skipif(
        not pytest.importorskip("transformers", reason="transformers not installed"),
        reason="transformers required for mock test",
    )
    @patch("transformers.models.auto.modeling_auto.AutoModelForVision2Seq")
    @patch("transformers.models.auto.processing_auto.AutoProcessor")
    def test_initialization(self, mock_processor, mock_model):
        from src.vlm.copilot import VLMCopilot
        copilot = VLMCopilot(device="cpu")

        assert copilot.device == "cpu"
        assert copilot._model is None
        assert copilot._processor is None

        # Trigger lazy loading
        _ = copilot.model
        assert mock_model.from_pretrained.called
        assert mock_processor.from_pretrained.called

    def test_assess_quality_parsing(self):
        from src.vlm.copilot import VLMCopilot
        copilot = VLMCopilot()

        # Mock the query response
        copilot.query_with_image = MagicMock(return_value="Defect detected. Severity is critical.")

        assessment = copilot.assess_quality(b"mock_image")

        assert assessment["defect_found"] is True
        assert assessment["severity"] == "critical"


class TestReportGenerator:
    """Test automated report generation."""

    def test_generate_from_bytes(self):
        from src.vlm.report_gen import ReportGenerator

        mock_copilot = MagicMock()
        mock_copilot.assess_quality.return_value = {
            "defect_found": True,
            "severity": "high",
            "raw_assessment": "Found deep scratch on surface."
        }
        mock_copilot.model_name = "mock_model"

        generator = ReportGenerator(mock_copilot)
        report = generator.generate_from_bytes(b"mock_image")

        assert report["pass_fail"] == "FAIL"
        assert report["severity"] == "high"
        assert "RPT-" in report["report_id"]

    def test_to_html(self):
        from src.vlm.report_gen import ReportGenerator
        generator = ReportGenerator(MagicMock())

        report = {
            "report_id": "RPT-123",
            "timestamp": "2026-05-09T00:00:00Z",
            "assessment": {"raw_assessment": "All good"},
            "pass_fail": "PASS",
            "severity": "none"
        }

        html = generator.to_html(report)
        assert "RPT-123" in html
        assert "PASS" in html
        assert "#6c757d" in html # 'none' severity color
