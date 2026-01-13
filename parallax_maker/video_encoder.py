# (c) 2024 Niels Provos
#
# Video encoding for camera animations.
#

from pathlib import Path
from typing import List, Optional, Callable, TYPE_CHECKING
import cv2
import numpy as np
from PIL import Image as PILImage

if TYPE_CHECKING:
    from .camera import Camera
    from .animation import CameraAnimation, AnimationConfig
    from .slice import ImageSlice
    from .effects import EffectsConfig


class VideoEncoder:
    """Encodes frames to video using OpenCV"""

    CODECS = {
        "mp4": ("mp4v", ".mp4"),
        "h264": ("avc1", ".mp4"),
        "avi": ("XVID", ".avi"),
    }

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: int = 30,
        codec: str = "mp4",
    ):
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps

        fourcc_code, extension = self.CODECS.get(codec, self.CODECS["mp4"])
        if self.output_path.suffix != extension:
            self.output_path = self.output_path.with_suffix(extension)

        self.fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        self.writer: Optional[cv2.VideoWriter] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Initialize the video writer"""
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            self.fourcc,
            self.fps,
            (self.width, self.height)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self.output_path}")

    def write_frame(self, frame: np.ndarray):
        """Write a single frame (expects RGB or RGBA)"""
        if self.writer is None:
            raise RuntimeError("Video writer not opened")

        # Convert RGBA to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Ensure correct size
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))

        self.writer.write(frame)

    def close(self):
        """Release the video writer"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def render_animation_to_video(
    output_path: Path,
    image_slices: List["ImageSlice"],
    card_corners_3d_list: List,
    camera: "Camera",
    animation: "CameraAnimation",
    config: "AnimationConfig",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    effects_config: Optional["EffectsConfig"] = None,
    depth_map: Optional[np.ndarray] = None,
) -> Path:
    """
    Render animation directly to video file.

    Args:
        output_path: Directory for output
        image_slices: List of ImageSlice objects
        card_corners_3d_list: Pre-computed 3D card corners
        camera: Camera object for projection matrix
        animation: CameraAnimation defining the movement
        config: AnimationConfig with export settings
        progress_callback: Optional callback(current, total)
        effects_config: Optional post-processing effects configuration
        depth_map: Optional depth map for DOF effect

    Returns:
        Path to the generated video file
    """
    from .segmentation import render_view
    from .camera import Camera as CameraClass
    from .effects import apply_effects, EffectsConfig

    if not image_slices:
        raise ValueError("No image slices to render")

    # Get image dimensions from first slice
    height, width = image_slices[0].image.shape[:2]

    # Determine output file
    video_path = Path(output_path) / f"animation.{config.output_format}"

    # Use default effects if none provided
    if effects_config is None:
        effects_config = EffectsConfig()

    # Create video encoder
    with VideoEncoder(
        video_path,
        width=width,
        height=height,
        fps=config.fps,
        codec="mp4" if config.output_format == "mp4" else "avi",
    ) as encoder:

        for frame_idx in range(config.num_frames):
            # Get interpolated camera state
            position, focal_length = animation.interpolate(frame_idx)

            # Update camera matrix with new focal length if changed
            temp_camera = CameraClass(
                distance=camera.camera_distance,
                max_distance=camera.max_distance,
                focal_length=focal_length,
            )
            camera_matrix = temp_camera.camera_matrix(width, height)

            # Render the frame
            frame = render_view(
                image_slices,
                camera_matrix,
                card_corners_3d_list,
                position,
            )

            # Apply post-processing effects
            frame = apply_effects(
                frame,
                effects_config,
                depth_map=depth_map,
                frame_index=frame_idx
            )

            # Write to video
            encoder.write_frame(frame)

            if progress_callback:
                progress_callback(frame_idx + 1, config.num_frames)

    return encoder.output_path


def render_animation_to_gif(
    output_path: Path,
    image_slices: List["ImageSlice"],
    card_corners_3d_list: List,
    camera: "Camera",
    animation: "CameraAnimation",
    config: "AnimationConfig",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    effects_config: Optional["EffectsConfig"] = None,
    depth_map: Optional[np.ndarray] = None,
) -> Path:
    """
    Render animation to GIF using PIL/Pillow.

    Args:
        output_path: Directory for output
        image_slices: List of ImageSlice objects
        card_corners_3d_list: Pre-computed 3D card corners
        camera: Camera object for projection matrix
        animation: CameraAnimation defining the movement
        config: AnimationConfig with export settings
        progress_callback: Optional callback(current, total)
        effects_config: Optional post-processing effects configuration
        depth_map: Optional depth map for DOF effect

    Returns:
        Path to the generated GIF file
    """
    from .segmentation import render_view
    from .camera import Camera as CameraClass
    from .effects import apply_effects, EffectsConfig

    if not image_slices:
        raise ValueError("No image slices to render")

    height, width = image_slices[0].image.shape[:2]
    gif_path = Path(output_path) / "animation.gif"

    if effects_config is None:
        effects_config = EffectsConfig()

    frames = []

    for frame_idx in range(config.num_frames):
        position, focal_length = animation.interpolate(frame_idx)

        temp_camera = CameraClass(
            distance=camera.camera_distance,
            max_distance=camera.max_distance,
            focal_length=focal_length,
        )
        camera_matrix = temp_camera.camera_matrix(width, height)

        frame = render_view(
            image_slices,
            camera_matrix,
            card_corners_3d_list,
            position,
        )

        # Apply post-processing effects
        frame = apply_effects(
            frame,
            effects_config,
            depth_map=depth_map,
            frame_index=frame_idx
        )

        # Convert RGBA to RGB for GIF
        pil_frame = PILImage.fromarray(frame[:, :, :3])
        frames.append(pil_frame)

        if progress_callback:
            progress_callback(frame_idx + 1, config.num_frames)

    # Calculate frame duration in milliseconds
    duration = int(1000 / config.fps)

    # Save as GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )

    return gif_path


def render_animation_to_sequence(
    output_path: Path,
    image_slices: List["ImageSlice"],
    card_corners_3d_list: List,
    camera: "Camera",
    animation: "CameraAnimation",
    config: "AnimationConfig",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    effects_config: Optional["EffectsConfig"] = None,
    depth_map: Optional[np.ndarray] = None,
) -> Path:
    """
    Render animation to PNG sequence.

    Args:
        output_path: Directory for output
        image_slices: List of ImageSlice objects
        card_corners_3d_list: Pre-computed 3D card corners
        camera: Camera object for projection matrix
        animation: CameraAnimation defining the movement
        config: AnimationConfig with export settings
        progress_callback: Optional callback(current, total)
        effects_config: Optional post-processing effects configuration
        depth_map: Optional depth map for DOF effect

    Returns:
        Path to the output directory
    """
    from .segmentation import render_view
    from .camera import Camera as CameraClass
    from .effects import apply_effects, EffectsConfig

    if not image_slices:
        raise ValueError("No image slices to render")

    height, width = image_slices[0].image.shape[:2]
    output_dir = Path(output_path)

    if effects_config is None:
        effects_config = EffectsConfig()

    for frame_idx in range(config.num_frames):
        position, focal_length = animation.interpolate(frame_idx)

        temp_camera = CameraClass(
            distance=camera.camera_distance,
            max_distance=camera.max_distance,
            focal_length=focal_length,
        )
        camera_matrix = temp_camera.camera_matrix(width, height)

        frame = render_view(
            image_slices,
            camera_matrix,
            card_corners_3d_list,
            position,
        )

        # Apply post-processing effects
        frame = apply_effects(
            frame,
            effects_config,
            depth_map=depth_map,
            frame_index=frame_idx
        )

        image_name = f"rendered_image_{frame_idx:03d}.png"
        output_image_path = output_dir / image_name

        cv2.imwrite(
            str(output_image_path), cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        )

        if progress_callback:
            progress_callback(frame_idx + 1, config.num_frames)

    return output_dir
