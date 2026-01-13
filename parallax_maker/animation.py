# (c) 2024 Niels Provos
#
# Camera animation system for 2.5D parallax effects.
#

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from .easing import EasingType, apply_easing

if TYPE_CHECKING:
    from .camera import Camera


class AnimationPreset(Enum):
    """Predefined camera animation presets"""
    NONE = "none"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    ORBIT_LEFT = "orbit_left"
    ORBIT_RIGHT = "orbit_right"
    KEN_BURNS = "ken_burns"


@dataclass
class CameraKeyframe:
    """Represents a camera state at a specific time"""
    position: np.ndarray
    focal_length: float
    time: float  # Normalized time (0.0 to 1.0)

    def __post_init__(self):
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)


@dataclass
class CameraAnimation:
    """Defines a complete camera animation"""
    name: str
    keyframes: List[CameraKeyframe] = field(default_factory=list)
    easing: EasingType = EasingType.EASE_IN_OUT
    duration_frames: int = 100

    def interpolate(self, frame: int) -> Tuple[np.ndarray, float]:
        """
        Interpolate camera state at given frame.

        Args:
            frame: The frame number to interpolate

        Returns:
            Tuple of (position, focal_length)
        """
        if not self.keyframes:
            raise ValueError("Animation has no keyframes")

        # Normalize frame to 0-1 range
        t = frame / max(self.duration_frames - 1, 1)
        t = np.clip(t, 0.0, 1.0)

        # Apply easing
        eased_t = apply_easing(t, self.easing)

        # Find surrounding keyframes
        prev_kf = self.keyframes[0]
        next_kf = self.keyframes[-1]

        for i, kf in enumerate(self.keyframes):
            if kf.time >= eased_t:
                next_kf = kf
                if i > 0:
                    prev_kf = self.keyframes[i - 1]
                break
            prev_kf = kf

        # Interpolate between keyframes
        if prev_kf.time == next_kf.time:
            local_t = 0.0
        else:
            local_t = (eased_t - prev_kf.time) / (next_kf.time - prev_kf.time)

        position = prev_kf.position + local_t * (next_kf.position - prev_kf.position)
        focal_length = prev_kf.focal_length + local_t * (next_kf.focal_length - prev_kf.focal_length)

        return position.astype(np.float32), focal_length


@dataclass
class AnimationConfig:
    """Configuration for animation export"""
    preset: AnimationPreset = AnimationPreset.DOLLY_IN
    easing: EasingType = EasingType.EASE_IN_OUT
    num_frames: int = 100
    fps: int = 30
    intensity: float = 1.0
    output_format: str = "mp4"
    quality: int = 23  # CRF for H.264 (lower = better quality)

    def to_dict(self) -> dict:
        return {
            "preset": self.preset.value,
            "easing": self.easing.value,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "intensity": self.intensity,
            "output_format": self.output_format,
            "quality": self.quality,
        }

    @staticmethod
    def from_dict(data: dict) -> "AnimationConfig":
        return AnimationConfig(
            preset=AnimationPreset(data.get("preset", "dolly_in")),
            easing=EasingType(data.get("easing", "ease_in_out")),
            num_frames=data.get("num_frames", 100),
            fps=data.get("fps", 30),
            intensity=data.get("intensity", 1.0),
            output_format=data.get("output_format", "mp4"),
            quality=data.get("quality", 23),
        )


class AnimationPresetFactory:
    """Factory for creating camera animations from presets"""

    @staticmethod
    def create(
        preset: AnimationPreset,
        camera: "Camera",
        image_width: int,
        image_height: int,
        config: AnimationConfig
    ) -> CameraAnimation:
        """
        Create a CameraAnimation from a preset.

        Args:
            preset: The animation preset to use
            camera: The camera object with initial settings
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            config: Animation configuration

        Returns:
            CameraAnimation instance ready for rendering
        """
        initial_pos = camera.camera_position.copy()
        focal_length = camera.focal_length
        intensity = config.intensity

        # Base movement amounts (scaled by intensity)
        pan_amount = image_width * 0.15 * intensity
        tilt_amount = image_height * 0.15 * intensity
        dolly_amount = camera.camera_distance * 0.5 * intensity
        zoom_factor = 0.3 * intensity
        orbit_angle = np.pi / 6 * intensity  # 30 degrees

        keyframes = []

        if preset == AnimationPreset.NONE:
            keyframes = [
                CameraKeyframe(initial_pos.copy(), focal_length, 0.0),
                CameraKeyframe(initial_pos.copy(), focal_length, 1.0),
            ]

        elif preset == AnimationPreset.PAN_LEFT:
            start_pos = initial_pos + np.array([pan_amount / 2, 0, 0], dtype=np.float32)
            end_pos = initial_pos - np.array([pan_amount / 2, 0, 0], dtype=np.float32)
            keyframes = [
                CameraKeyframe(start_pos, focal_length, 0.0),
                CameraKeyframe(end_pos, focal_length, 1.0),
            ]

        elif preset == AnimationPreset.PAN_RIGHT:
            start_pos = initial_pos - np.array([pan_amount / 2, 0, 0], dtype=np.float32)
            end_pos = initial_pos + np.array([pan_amount / 2, 0, 0], dtype=np.float32)
            keyframes = [
                CameraKeyframe(start_pos, focal_length, 0.0),
                CameraKeyframe(end_pos, focal_length, 1.0),
            ]

        elif preset == AnimationPreset.TILT_UP:
            start_pos = initial_pos + np.array([0, tilt_amount / 2, 0], dtype=np.float32)
            end_pos = initial_pos - np.array([0, tilt_amount / 2, 0], dtype=np.float32)
            keyframes = [
                CameraKeyframe(start_pos, focal_length, 0.0),
                CameraKeyframe(end_pos, focal_length, 1.0),
            ]

        elif preset == AnimationPreset.TILT_DOWN:
            start_pos = initial_pos - np.array([0, tilt_amount / 2, 0], dtype=np.float32)
            end_pos = initial_pos + np.array([0, tilt_amount / 2, 0], dtype=np.float32)
            keyframes = [
                CameraKeyframe(start_pos, focal_length, 0.0),
                CameraKeyframe(end_pos, focal_length, 1.0),
            ]

        elif preset == AnimationPreset.DOLLY_IN:
            start_pos = initial_pos.copy()
            end_pos = initial_pos + np.array([0, 0, dolly_amount], dtype=np.float32)
            keyframes = [
                CameraKeyframe(start_pos, focal_length, 0.0),
                CameraKeyframe(end_pos, focal_length, 1.0),
            ]

        elif preset == AnimationPreset.DOLLY_OUT:
            start_pos = initial_pos + np.array([0, 0, dolly_amount], dtype=np.float32)
            end_pos = initial_pos.copy()
            keyframes = [
                CameraKeyframe(start_pos, focal_length, 0.0),
                CameraKeyframe(end_pos, focal_length, 1.0),
            ]

        elif preset == AnimationPreset.ZOOM_IN:
            keyframes = [
                CameraKeyframe(initial_pos.copy(), focal_length, 0.0),
                CameraKeyframe(initial_pos.copy(), focal_length * (1 + zoom_factor), 1.0),
            ]

        elif preset == AnimationPreset.ZOOM_OUT:
            keyframes = [
                CameraKeyframe(initial_pos.copy(), focal_length * (1 + zoom_factor), 0.0),
                CameraKeyframe(initial_pos.copy(), focal_length, 1.0),
            ]

        elif preset == AnimationPreset.ORBIT_LEFT:
            keyframes = AnimationPresetFactory._create_orbit_keyframes(
                initial_pos, focal_length, camera.camera_distance * 0.3 * intensity,
                start_angle=0, end_angle=orbit_angle, num_keyframes=10
            )

        elif preset == AnimationPreset.ORBIT_RIGHT:
            keyframes = AnimationPresetFactory._create_orbit_keyframes(
                initial_pos, focal_length, camera.camera_distance * 0.3 * intensity,
                start_angle=0, end_angle=-orbit_angle, num_keyframes=10
            )

        elif preset == AnimationPreset.KEN_BURNS:
            # Slow dolly in with subtle pan and slight zoom
            mid_pos = initial_pos + np.array(
                [pan_amount * 0.1, 0, dolly_amount * 0.5], dtype=np.float32
            )
            end_pos = initial_pos + np.array(
                [pan_amount * 0.2, 0, dolly_amount], dtype=np.float32
            )
            keyframes = [
                CameraKeyframe(initial_pos.copy(), focal_length, 0.0),
                CameraKeyframe(mid_pos, focal_length * 1.05, 0.5),
                CameraKeyframe(end_pos, focal_length * 1.1, 1.0),
            ]

        return CameraAnimation(
            name=preset.value,
            keyframes=keyframes,
            easing=config.easing,
            duration_frames=config.num_frames,
        )

    @staticmethod
    def _create_orbit_keyframes(
        center: np.ndarray,
        focal_length: float,
        radius: float,
        start_angle: float,
        end_angle: float,
        num_keyframes: int
    ) -> List[CameraKeyframe]:
        """Create keyframes for orbital camera movement"""
        keyframes = []
        for i in range(num_keyframes):
            t = i / (num_keyframes - 1)
            angle = start_angle + t * (end_angle - start_angle)

            # Orbit in XZ plane around the center
            offset = np.array([
                radius * np.sin(angle),
                0,
                radius * (1 - np.cos(angle))
            ], dtype=np.float32)

            pos = center + offset
            keyframes.append(CameraKeyframe(pos, focal_length, t))

        return keyframes
