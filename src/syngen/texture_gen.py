"""
Texture Generator — Diffusion-based Defect Texture Synthesis
==============================================================
Uses Stable Diffusion XL + ControlNet for photorealistic
industrial defect texture generation.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger


class TextureGenerator:
    """Generate photorealistic defect textures using Stable Diffusion XL.

    Uses diffusion models to create high-quality surface defect textures
    that can be applied to 3D models for synthetic training data.

    Args:
        model_id: HuggingFace model ID for the diffusion model.
        output_dir: Directory to save generated textures.
        device: Compute device.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        output_dir: str | Path = "data/synthetic/textures",
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None
        self._gen_count = 0

    @property
    def pipe(self):
        """Lazy-load the diffusion pipeline."""
        if self._pipe is None:
            from diffusers import StableDiffusionXLPipeline

            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device == "cuda" else None,
            )
            self._pipe = self._pipe.to(self.device)

            # Optimize memory
            if self.device == "cuda":
                self._pipe.enable_model_cpu_offload()

            logger.info(f"Diffusion pipeline loaded: {self.model_id}")

        return self._pipe

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality, watermark, text",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ) -> Path:
        """Generate a single defect texture image.

        Args:
            prompt: Text description of the defect texture.
            negative_prompt: What to avoid in generation.
            width: Output image width.
            height: Output image height.
            num_inference_steps: Denoising steps (more = higher quality).
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.

        Returns:
            Path to the generated texture image.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        # Save texture
        self._gen_count += 1
        filename = f"texture_{self._gen_count:05d}.png"
        save_path = self.output_dir / filename
        image.save(str(save_path))

        return save_path

    def generate_batch(
        self,
        prompts: list[str],
        **kwargs,
    ) -> list[Path]:
        """Generate a batch of textures from multiple prompts."""
        paths = []
        for i, prompt in enumerate(prompts):
            seed = kwargs.get("seed", 42) + i
            path = self.generate(prompt=prompt, seed=seed, **{k: v for k, v in kwargs.items() if k != "seed"})
            paths.append(path)
        return paths
