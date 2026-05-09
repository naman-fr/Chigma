"""
Scene Renderer — 3D Industrial Scene Generation
=================================================
Renders synthetic industrial scenes with defect textures applied
to 3D metal surfaces. Supports randomized lighting, camera angles,
and background environments.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import trimesh
    from PIL import Image
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


class SceneRenderer:
    """Render 3D industrial scenes with synthetic defect textures.

    Creates photorealistic training images by:
    1. Creating a flat metal plate mesh
    2. Applying generated defect textures via UV mapping
    3. Randomizing lighting conditions and camera viewpoints
    4. Rendering to images with known defect locations

    Args:
        output_dir: Directory for rendered images.
        image_size: Output image dimensions (square).
    """

    def __init__(
        self,
        output_dir: str | Path = "data/synthetic/images",
        image_size: int = 640,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self._render_count = 0

    def render(
        self,
        texture_path: str | Path,
        defect_type: str,
        image_index: int = 0,
        lighting_var: int = 0,
        camera_angle: int = 0,
    ) -> dict[str, Any]:
        """Render a single scene with defect texture.

        Args:
            texture_path: Path to defect texture image.
            defect_type: Type of defect being rendered.
            image_index: Index for the output filename.
            lighting_var: Lighting variation index.
            camera_angle: Camera angle variation index.

        Returns:
            Scene metadata dict with image path and defect locations.
        """
        texture_path = Path(texture_path)
        self._render_count += 1

        # Generate the scene image
        scene_image, defect_regions = self._compose_scene(
            texture_path, lighting_var, camera_angle
        )

        # Save rendered image
        filename = f"{defect_type}_{image_index:05d}.png"
        save_path = self.output_dir / filename
        scene_image.save(str(save_path))

        metadata = {
            "image_path": str(save_path),
            "defect_type": defect_type,
            "defect_regions": defect_regions,
            "image_size": self.image_size,
            "lighting_var": lighting_var,
            "camera_angle": camera_angle,
        }

        return metadata

    def _compose_scene(
        self,
        texture_path: Path,
        lighting_var: int,
        camera_angle: int,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Compose a synthetic scene by overlaying defect texture on background.

        Uses PIL-based compositing as a lightweight alternative to full 3D
        rendering. For production, integrate with Blender via bpy.

        Returns:
            Tuple of (rendered PIL image, list of defect region dicts).
        """
        from PIL import Image

        # Create base metal background with texture variation
        bg = self._generate_metal_background(lighting_var)

        # Load and prepare defect texture
        texture = Image.open(texture_path).convert("RGBA")

        # Randomize defect placement
        defect_regions = []
        n_defects = random.randint(1, 3)

        for _ in range(n_defects):
            # Random size (10-40% of image)
            scale = random.uniform(0.1, 0.4)
            w = int(self.image_size * scale)
            h = int(self.image_size * scale * random.uniform(0.5, 1.5))
            h = min(h, self.image_size)

            # Random position
            x = random.randint(0, max(0, self.image_size - w))
            y = random.randint(0, max(0, self.image_size - h))

            # Random rotation based on camera angle
            rotation = camera_angle * 45 + random.uniform(-15, 15)

            # Resize and rotate defect texture
            defect = texture.resize((w, h), Image.Resampling.LANCZOS)
            defect = defect.rotate(rotation, expand=True, resample=Image.Resampling.BILINEAR)

            # Apply alpha blending for realistic overlay
            alpha = random.uniform(0.4, 0.9)
            defect_alpha = defect.split()[-1]
            defect_alpha = defect_alpha.point(lambda p: int(p * alpha))
            defect.putalpha(defect_alpha)

            # Paste defect onto background
            paste_x = min(x, self.image_size - defect.width)
            paste_y = min(y, self.image_size - defect.height)
            paste_x = max(0, paste_x)
            paste_y = max(0, paste_y)

            bg.paste(defect, (paste_x, paste_y), defect)

            # Record defect region for labeling (YOLO format: normalized xywh)
            cx = (paste_x + defect.width / 2) / self.image_size
            cy = (paste_y + defect.height / 2) / self.image_size
            rw = defect.width / self.image_size
            rh = defect.height / self.image_size

            defect_regions.append({
                "cx": round(min(cx, 1.0), 6),
                "cy": round(min(cy, 1.0), 6),
                "w": round(min(rw, 1.0), 6),
                "h": round(min(rh, 1.0), 6),
            })

        # Apply post-processing for realism
        bg = self._apply_post_processing(bg, lighting_var)

        return bg, defect_regions

    def _generate_metal_background(self, lighting_var: int) -> Any:
        """Generate a synthetic metal surface background."""
        from PIL import Image

        bg = Image.new("RGBA", (self.image_size, self.image_size))

        # Metal-like base color with variation
        base_colors = [
            (160, 165, 170),  # Cool steel
            (170, 168, 160),  # Warm steel
            (150, 155, 160),  # Dark steel
            (180, 178, 175),  # Light steel
            (155, 160, 165),  # Blue steel
        ]
        base = base_colors[lighting_var % len(base_colors)]

        # Add noise for metal texture
        pixels = np.random.normal(0, 8, (self.image_size, self.image_size, 3))
        metal = np.clip(np.array(base) + pixels, 0, 255).astype(np.uint8)

        # Add directional grain (rolling direction)
        for y in range(self.image_size):
            offset = np.random.randint(-3, 4)
            metal[y, :, :] = np.clip(metal[y, :, :].astype(int) + offset, 0, 255)

        bg = Image.fromarray(metal, "RGB").convert("RGBA")
        return bg

    def _apply_post_processing(self, image: Any, lighting_var: int) -> Any:
        """Apply post-processing effects for photorealism."""
        from PIL import ImageEnhance, ImageFilter

        image = image.convert("RGB")

        # Slight blur for realism
        image = image.filter(ImageFilter.GaussianBlur(radius=0.3))

        # Brightness/contrast variation based on lighting
        brightness_factor = 0.8 + (lighting_var % 5) * 0.1
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)

        contrast_factor = 0.9 + random.uniform(-0.1, 0.2)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        return image
