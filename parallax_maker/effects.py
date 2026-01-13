# (c) 2024 Niels Provos
#
# Post-processing effects for parallax animations.
# Inspired by DepthFlow's shader effects pipeline.
#

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2


@dataclass
class EffectsConfig:
    """Configuration for post-processing effects"""
    # Vignette
    vignette_enabled: bool = False
    vignette_intensity: float = 0.3  # 0.0 to 1.0
    vignette_radius: float = 0.8  # 0.0 to 1.5

    # Depth of Field
    dof_enabled: bool = False
    dof_focus_depth: float = 0.5  # 0.0 (near) to 1.0 (far)
    dof_blur_amount: float = 15.0  # Blur kernel size
    dof_focus_range: float = 0.2  # Range around focus that stays sharp

    # Lens Distortion
    lens_distortion_enabled: bool = False
    lens_distortion_k1: float = 0.0  # Radial distortion coefficient
    lens_distortion_k2: float = 0.0  # Secondary radial coefficient

    # Chromatic Aberration
    chromatic_aberration_enabled: bool = False
    chromatic_aberration_strength: float = 2.0  # Pixel offset

    # Film Grain
    grain_enabled: bool = False
    grain_intensity: float = 0.05  # 0.0 to 0.2

    def to_dict(self) -> dict:
        return {
            "vignette_enabled": self.vignette_enabled,
            "vignette_intensity": self.vignette_intensity,
            "vignette_radius": self.vignette_radius,
            "dof_enabled": self.dof_enabled,
            "dof_focus_depth": self.dof_focus_depth,
            "dof_blur_amount": self.dof_blur_amount,
            "dof_focus_range": self.dof_focus_range,
            "lens_distortion_enabled": self.lens_distortion_enabled,
            "lens_distortion_k1": self.lens_distortion_k1,
            "lens_distortion_k2": self.lens_distortion_k2,
            "chromatic_aberration_enabled": self.chromatic_aberration_enabled,
            "chromatic_aberration_strength": self.chromatic_aberration_strength,
            "grain_enabled": self.grain_enabled,
            "grain_intensity": self.grain_intensity,
        }

    @staticmethod
    def from_dict(data: dict) -> "EffectsConfig":
        return EffectsConfig(**{k: v for k, v in data.items() if k in EffectsConfig.__dataclass_fields__})


def apply_vignette(
    image: np.ndarray,
    intensity: float = 0.3,
    radius: float = 0.8
) -> np.ndarray:
    """
    Apply vignette effect (darken edges).

    Args:
        image: Input image (RGB or RGBA)
        intensity: Strength of darkening (0.0 to 1.0)
        radius: Size of the bright center (0.0 to 1.5)

    Returns:
        Image with vignette applied
    """
    height, width = image.shape[:2]

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Normalize to -1 to 1 range
    center_x, center_y = width / 2, height / 2
    x_norm = (x - center_x) / center_x
    y_norm = (y - center_y) / center_y

    # Calculate distance from center
    distance = np.sqrt(x_norm**2 + y_norm**2)

    # Create vignette mask using smooth falloff
    vignette = 1.0 - np.clip((distance - radius) / (1.0 - radius + 0.001), 0, 1) * intensity
    vignette = np.power(vignette, 1.5)  # Smooth the falloff

    # Apply to image
    result = image.copy().astype(np.float32)
    for i in range(min(3, image.shape[2])):
        result[:, :, i] = result[:, :, i] * vignette

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_depth_of_field(
    image: np.ndarray,
    depth_map: np.ndarray,
    focus_depth: float = 0.5,
    blur_amount: float = 15.0,
    focus_range: float = 0.2
) -> np.ndarray:
    """
    Apply depth-based blur (depth of field).

    Args:
        image: Input image (RGB or RGBA)
        depth_map: Depth map (grayscale, 0-255)
        focus_depth: Depth value to focus on (0.0 to 1.0)
        blur_amount: Maximum blur kernel size
        focus_range: Range around focus_depth that stays sharp

    Returns:
        Image with depth of field applied
    """
    height, width = image.shape[:2]

    # Normalize depth map
    if depth_map.ndim == 3:
        depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
    depth_norm = depth_map.astype(np.float32) / 255.0

    # Calculate blur amount per pixel based on distance from focus
    focus_distance = np.abs(depth_norm - focus_depth)
    blur_mask = np.clip((focus_distance - focus_range) / (1.0 - focus_range + 0.001), 0, 1)

    # Create multiple blur levels
    blur_levels = []
    kernel_sizes = [3, 7, 15, 25, 35]
    for k in kernel_sizes:
        if k <= blur_amount:
            blurred = cv2.GaussianBlur(image, (k, k), 0)
            blur_levels.append(blurred)

    if not blur_levels:
        return image

    # Blend between blur levels based on blur_mask
    result = image.copy().astype(np.float32)
    num_levels = len(blur_levels)

    for y in range(height):
        for level_idx in range(num_levels):
            level_start = level_idx / num_levels
            level_end = (level_idx + 1) / num_levels

            mask_row = blur_mask[y, :]
            in_level = (mask_row >= level_start) & (mask_row < level_end)

            if np.any(in_level):
                local_t = (mask_row[in_level] - level_start) / (level_end - level_start + 0.001)

                if level_idx < num_levels - 1:
                    for c in range(min(3, image.shape[2])):
                        result[y, in_level, c] = (
                            blur_levels[level_idx][y, in_level, c] * (1 - local_t) +
                            blur_levels[level_idx + 1][y, in_level, c] * local_t
                        )
                else:
                    result[y, in_level, :3] = blur_levels[level_idx][y, in_level, :3]

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_lens_distortion(
    image: np.ndarray,
    k1: float = 0.1,
    k2: float = 0.0
) -> np.ndarray:
    """
    Apply barrel/pincushion lens distortion.

    Args:
        image: Input image
        k1: Primary radial distortion (positive = barrel, negative = pincushion)
        k2: Secondary radial distortion coefficient

    Returns:
        Distorted image
    """
    height, width = image.shape[:2]

    # Camera matrix (assume center of image)
    fx = fy = width
    cx, cy = width / 2, height / 2
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # Distortion coefficients
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )

    # Apply undistortion (which with positive k1 creates barrel distortion)
    result = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    return result


def apply_chromatic_aberration(
    image: np.ndarray,
    strength: float = 2.0
) -> np.ndarray:
    """
    Apply chromatic aberration (RGB channel separation at edges).

    Args:
        image: Input image (RGB or RGBA)
        strength: Pixel offset for color channels

    Returns:
        Image with chromatic aberration
    """
    height, width = image.shape[:2]
    has_alpha = image.shape[2] == 4

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    center_x, center_y = width / 2, height / 2

    # Calculate radial distance from center
    x_offset = (x - center_x) / center_x
    y_offset = (y - center_y) / center_y

    # Offset red channel outward, blue channel inward
    red_offset_x = (x_offset * strength).astype(np.float32)
    red_offset_y = (y_offset * strength).astype(np.float32)

    blue_offset_x = (-x_offset * strength).astype(np.float32)
    blue_offset_y = (-y_offset * strength).astype(np.float32)

    # Create mapping grids
    map_x = np.arange(width, dtype=np.float32)
    map_y = np.arange(height, dtype=np.float32)
    map_x, map_y = np.meshgrid(map_x, map_y)

    # Remap each channel
    result = image.copy()

    # Red channel - shift outward
    red_map_x = np.clip(map_x + red_offset_x, 0, width - 1).astype(np.float32)
    red_map_y = np.clip(map_y + red_offset_y, 0, height - 1).astype(np.float32)
    result[:, :, 0] = cv2.remap(image[:, :, 0], red_map_x, red_map_y, cv2.INTER_LINEAR)

    # Blue channel - shift inward
    blue_map_x = np.clip(map_x + blue_offset_x, 0, width - 1).astype(np.float32)
    blue_map_y = np.clip(map_y + blue_offset_y, 0, height - 1).astype(np.float32)
    result[:, :, 2] = cv2.remap(image[:, :, 2], blue_map_x, blue_map_y, cv2.INTER_LINEAR)

    return result


def apply_film_grain(
    image: np.ndarray,
    intensity: float = 0.05,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add film grain noise.

    Args:
        image: Input image
        intensity: Grain strength (0.0 to 0.2)
        seed: Random seed for reproducible grain

    Returns:
        Image with film grain
    """
    if seed is not None:
        np.random.seed(seed)

    height, width = image.shape[:2]

    # Generate gaussian noise
    noise = np.random.normal(0, intensity * 255, (height, width))

    # Apply to all color channels
    result = image.copy().astype(np.float32)
    for i in range(min(3, image.shape[2])):
        result[:, :, i] = result[:, :, i] + noise

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_effects(
    image: np.ndarray,
    config: EffectsConfig,
    depth_map: Optional[np.ndarray] = None,
    frame_index: int = 0
) -> np.ndarray:
    """
    Apply all enabled post-processing effects.

    Args:
        image: Input image (RGB or RGBA)
        config: Effects configuration
        depth_map: Optional depth map for DOF effect
        frame_index: Current frame (for grain seed)

    Returns:
        Processed image
    """
    result = image.copy()

    # Apply effects in order (order matters!)

    # 1. Lens distortion (geometric, should be first)
    if config.lens_distortion_enabled and (config.lens_distortion_k1 != 0 or config.lens_distortion_k2 != 0):
        result = apply_lens_distortion(
            result,
            k1=config.lens_distortion_k1,
            k2=config.lens_distortion_k2
        )

    # 2. Depth of field (if depth map available)
    if config.dof_enabled and depth_map is not None:
        result = apply_depth_of_field(
            result,
            depth_map,
            focus_depth=config.dof_focus_depth,
            blur_amount=config.dof_blur_amount,
            focus_range=config.dof_focus_range
        )

    # 3. Chromatic aberration
    if config.chromatic_aberration_enabled and config.chromatic_aberration_strength > 0:
        result = apply_chromatic_aberration(
            result,
            strength=config.chromatic_aberration_strength
        )

    # 4. Vignette
    if config.vignette_enabled:
        result = apply_vignette(
            result,
            intensity=config.vignette_intensity,
            radius=config.vignette_radius
        )

    # 5. Film grain (last, as it's per-pixel noise)
    if config.grain_enabled and config.grain_intensity > 0:
        result = apply_film_grain(
            result,
            intensity=config.grain_intensity,
            seed=frame_index  # Different grain per frame
        )

    return result
