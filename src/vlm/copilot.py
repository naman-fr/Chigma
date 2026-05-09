"""
VLM Copilot — Vision-Language Model for Industrial Inspection
===============================================================
Natural language visual QA for defect analysis, quality assessment,
and automated reasoning about industrial images.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from PIL import Image


class VLMCopilot:
    """Vision-Language Model Copilot for industrial inspection.

    Provides natural language interface for:
    - Defect identification and classification
    - Quality assessment and pass/fail decisions
    - Severity rating and root cause analysis
    - Natural language queries about inspection images

    Args:
        model_name: HuggingFace model ID (default: Qwen2.5-VL-7B).
        device: Compute device.
        max_tokens: Maximum response length.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        max_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

        logger.info(f"VLM Copilot initialized: {model_name} on {self.device}")

    @property
    def model(self):
        """Lazy-load the VLM model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def processor(self):
        """Lazy-load the processor."""
        if self._processor is None:
            self._load_model()
        return self._processor

    def _load_model(self) -> None:
        """Load the VLM model and processor."""
        from transformers import AutoModelForVision2Seq, AutoProcessor

        logger.info(f"Loading VLM: {self.model_name}...")

        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device != "cuda":
            self._model = self._model.to(self.device)

        logger.info("VLM loaded successfully")

    def query_with_image(
        self,
        image: bytes | str | Path | Image.Image,
        query: str,
        system_prompt: str | None = None,
    ) -> str:
        """Query the VLM about an image.

        Args:
            image: Image data (bytes, path, or PIL Image).
            query: Natural language question.
            system_prompt: Optional system context.

        Returns:
            VLM text response.
        """
        # Convert to PIL Image
        if isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        if system_prompt is None:
            system_prompt = (
                "You are an expert industrial quality inspector specializing in "
                "steel surface defect detection. Analyze images for defects including: "
                "crazing, inclusion, patches, pitted surfaces, rolled-in scale, and scratches. "
                "Provide specific, actionable assessments."
            )

        # Build conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": query},
                ],
            },
        ]

        # Process and generate
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )

        # Decode response (skip input tokens)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response.strip()

    def assess_quality(self, image: bytes | str | Path) -> dict[str, Any]:
        """Perform structured quality assessment.

        Returns:
            Dict with defect_found, severity, defect_type, recommendation.
        """
        from src.vlm.prompt_templates import QUALITY_ASSESSMENT_PROMPT

        response = self.query_with_image(image, QUALITY_ASSESSMENT_PROMPT)

        # Parse structured response
        result: dict[str, Any] = {
            "raw_assessment": response,
            "defect_found": any(
                keyword in response.lower()
                for keyword in ["defect", "crack", "scratch", "pit", "inclusion", "patch"]
            ),
        }

        # Extract severity if mentioned
        for level in ["critical", "high", "medium", "low", "none"]:
            if level in response.lower():
                result["severity"] = level
                break
        else:
            result["severity"] = "unknown"

        return result
