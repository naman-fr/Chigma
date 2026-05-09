"""
SynGen-Vision Pipeline — Synthetic Defect Data Generation
==========================================================
End-to-end pipeline that generates photorealistic, auto-labeled
defect images using Stable Diffusion + VLM prompt generation + 3D rendering.

Reference: "Synthetic Data Generation for Training Industrial Vision Models" (2025)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
import click
from loguru import logger


class SynGenPipeline:
    """SynGen-Vision Orchestrator.

    Pipeline stages:
    1. VLM generates diverse defect texture descriptions
    2. Diffusion model creates photorealistic defect textures
    3. 3D renderer places textures on industrial surfaces
    4. Auto-labeler extracts YOLO-format annotations

    Args:
        config: Pipeline configuration dictionary.
        output_dir: Directory for generated dataset.
    """

    DEFECT_TYPES = [
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled_in_scale", "scratches",
    ]

    def __init__(
        self,
        output_dir: str | Path = "data/synthetic",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or self._default_config()

        # Create output directories
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "textures").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metadata").mkdir(parents=True, exist_ok=True)

        logger.info(f"SynGen pipeline initialized | output={self.output_dir}")

    @staticmethod
    def _default_config() -> dict[str, Any]:
        """Default pipeline configuration."""
        return {
            "num_images_per_class": 500,
            "image_size": 640,
            "texture_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "controlnet": "lllyasviel/control_v11p_sd15_canny",
            "num_defects_per_image": [1, 4],  # min, max
            "backgrounds": ["steel_plate", "galvanized_surface", "rolled_metal"],
            "lighting_variations": 5,
            "camera_angles": 8,
            "train_val_split": 0.85,
        }

    def generate_prompts(self, defect_type: str, n_prompts: int = 10) -> list[str]:
        """Generate diverse defect texture prompts using VLM.

        Uses structured prompt templates for consistent, high-quality
        defect texture generation across different conditions.
        """
        base_prompts = {
            "crazing": [
                "fine network of hairline cracks on steel surface, industrial metal texture, high resolution macro photography",
                "spider web pattern of micro-fractures on polished steel, surface crazing defect, harsh industrial lighting",
                "interconnected fine cracks forming mosaic pattern on metal plate, thermal crazing, top-down view",
            ],
            "inclusion": [
                "dark non-metallic particle embedded in bright steel surface, slag inclusion defect, macro close-up",
                "irregular dark spot with sharp boundary on rolled steel, oxide inclusion, studio lighting",
                "small foreign material trapped in metal surface during rolling, inclusion defect, industrial inspection",
            ],
            "patches": [
                "irregular discolored area on steel surface, surface patch defect, uneven color distribution",
                "blotchy oxidation patch on galvanized steel, surface contamination, high contrast",
                "localized surface roughness change on metal plate, patch defect with texture variation",
            ],
            "pitted_surface": [
                "small round cavities and pits on steel surface, corrosion pitting, macro detail photography",
                "scattered micro-pits on polished metal surface, pitted surface defect, even lighting",
                "dimpled steel surface with shallow erosion marks, acid pitting, industrial quality inspection",
            ],
            "rolled_in_scale": [
                "dark oxide scale pressed into steel surface during hot rolling, flaky texture, industrial",
                "embedded mill scale fragments on metal surface, rolled-in scale defect, high resolution",
                "thin dark layers of iron oxide trapped under steel surface, scale inclusions, cross-lighting",
            ],
            "scratches": [
                "long linear mark on steel surface from mechanical contact, scratch defect, directional lighting",
                "parallel fine scratches on polished metal, handling damage, high contrast macro",
                "deep groove scratch across steel plate surface, mechanical surface damage, harsh lighting",
            ],
        }

        prompts = base_prompts.get(defect_type, [])
        # Extend with variations
        variations = [
            "harsh overhead lighting", "angled side lighting",
            "diffuse soft lighting", "high contrast",
            "slightly rusty background", "clean polished surface",
        ]

        extended = []
        for prompt in prompts:
            for var in variations[:n_prompts // len(prompts) + 1]:
                extended.append(f"{prompt}, {var}")

        return extended[:n_prompts]

    def generate_textures(
        self,
        defect_type: str,
        prompts: list[str],
    ) -> list[Path]:
        """Generate defect textures using Stable Diffusion XL.

        Args:
            defect_type: Type of defect to generate.
            prompts: Text prompts for texture generation.

        Returns:
            List of paths to generated texture images.
        """
        from src.syngen.texture_gen import TextureGenerator

        generator = TextureGenerator(
            model_id=self.config["texture_model"],
            output_dir=self.output_dir / "textures" / defect_type,
        )

        texture_paths = []
        for i, prompt in enumerate(prompts):
            path = generator.generate(
                prompt=prompt,
                negative_prompt="blurry, low quality, watermark, text, cartoon, painting",
                num_inference_steps=30,
                guidance_scale=7.5,
                seed=42 + i,
            )
            texture_paths.append(path)
            logger.debug(f"Generated texture {i + 1}/{len(prompts)} for {defect_type}")

        logger.info(f"Generated {len(texture_paths)} textures for {defect_type}")
        return texture_paths

    def render_scenes(
        self,
        defect_type: str,
        texture_paths: list[Path],
        n_images: int = 500,
    ) -> list[dict[str, Any]]:
        """Render 3D scenes with defect textures applied.

        Args:
            defect_type: Defect class name.
            texture_paths: Paths to generated textures.
            n_images: Number of scene images to render.

        Returns:
            List of scene metadata dicts (image path, annotations).
        """
        from src.syngen.scene_renderer import SceneRenderer

        renderer = SceneRenderer(output_dir=self.output_dir / "images")

        scenes = []
        for i in range(n_images):
            texture = texture_paths[i % len(texture_paths)]
            scene = renderer.render(
                texture_path=texture,
                defect_type=defect_type,
                image_index=i,
                lighting_var=i % self.config["lighting_variations"],
                camera_angle=i % self.config["camera_angles"],
            )
            scenes.append(scene)

        logger.info(f"Rendered {len(scenes)} scenes for {defect_type}")
        return scenes

    def auto_label(self, scenes: list[dict[str, Any]]) -> None:
        """Generate YOLO-format labels from scene metadata."""
        from src.syngen.auto_label import AutoLabeler

        labeler = AutoLabeler(output_dir=self.output_dir / "labels")
        labeler.generate_labels(scenes)
        logger.info(f"Generated labels for {len(scenes)} scenes")

    def run(self) -> dict[str, Any]:
        """Execute the complete SynGen-Vision pipeline.

        Returns:
            Pipeline statistics and output paths.
        """
        logger.info("Starting SynGen-Vision pipeline")
        stats: dict[str, Any] = {"defect_types": {}, "total_images": 0}

        for defect_type in self.DEFECT_TYPES:
            logger.info(f"Processing defect type: {defect_type}")

            # Stage 1: Generate prompts
            prompts = self.generate_prompts(defect_type, n_prompts=20)

            # Stage 2: Generate textures
            texture_paths = self.generate_textures(defect_type, prompts)

            # Stage 3: Render 3D scenes
            n_images = self.config["num_images_per_class"]
            scenes = self.render_scenes(defect_type, texture_paths, n_images)

            # Stage 4: Auto-label
            self.auto_label(scenes)

            stats["defect_types"][defect_type] = {
                "n_textures": len(texture_paths),
                "n_images": len(scenes),
            }
            stats["total_images"] += len(scenes)

        # Save pipeline metadata
        meta_path = self.output_dir / "metadata" / "pipeline_stats.json"
        with open(meta_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Pipeline complete | {stats['total_images']} images generated")
        return stats


@click.command()
@click.option("--output", default="data/synthetic", help="Output directory")
@click.option("--n-per-class", default=500, help="Images per defect class")
def main(output: str, n_per_class: int) -> None:
    """Run SynGen-Vision synthetic data generation pipeline."""
    config = SynGenPipeline._default_config()
    config["num_images_per_class"] = n_per_class
    pipeline = SynGenPipeline(output_dir=output, config=config)
    pipeline.run()


if __name__ == "__main__":
    main()
