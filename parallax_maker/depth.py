# (c) 2024 Niels Provos
#
"""
Create Depth Maps from Images

This module provides functionality to create depth maps from images using pre-trained deep learning models.
The depth maps can be used to create parallax effects in images and videos.

TODO:
 - Investigate https://huggingface.co/LiheYoung/depth_anything_vitl14 - also at https://huggingface.co/docs/transformers/main/model_doc/depth_anything
"""


from transformers import AutoImageProcessor, DPTForDepthEstimation, AutoModelForDepthEstimation
import torch
import cv2
import numpy as np
from PIL import Image

from .utils import torch_get_device


class DepthEstimationModel:
    MODELS = ["depth_anything_v2", "midas", "zoedepth", "dinov2"]

    def __init__(self, model="midas"):
        assert model in self.MODELS, f"Model {model} must be one of {self.MODELS}"
        self._model_name = model
        self.model = None
        self.transforms = None
        self.image_processor = None

    def __eq__(self, other):
        if not isinstance(other, DepthEstimationModel):
            return False
        return self._model_name == other._model_name

    @property
    def model_name(self):
        return self._model_name

    def load_model(self, progress_callback=None):
        load_pipeline = {
            "depth_anything_v2": create_depth_anything_v2_pipeline,
            "midas": create_medias_pipeline,
            "zoedepth": create_zoedepth_pipeline,
            "dinov2": create_dinov2_pipeline,
        }

        result = load_pipeline[self._model_name](progress_callback=progress_callback)

        if self._model_name == "midas":
            self.model, self.transforms = result
        elif self._model_name == "zoedepth":
            self.model = result
        elif self._model_name == "dinov2":
            self.model, self.image_processor = result
        elif self._model_name == "depth_anything_v2":
            self.model, self.image_processor = result

    def depth_map(self, image, progress_callback=None):
        if self.model is None:
            self.load_model()

        run_pipeline = {
            "depth_anything_v2": lambda img, cb: run_depth_anything_v2_pipeline(
                img, self.model, self.image_processor, progress_callback=cb
            ),
            "midas": lambda img, cb: run_medias_pipeline(
                img, self.model, self.transforms, progress_callback=cb
            ),
            "zoedepth": lambda img, cb: run_zoedepth_pipeline(
                img, self.model, progress_callback=cb
            ),
            "dinov2": lambda img, cb: run_dinov2_pipeline(
                img, self.model, self.image_processor, progress_callback=cb
            ),
        }

        return run_pipeline[self._model_name](image, progress_callback)


# =============================================================================
# Depth Anything V2 - State of the art (2025)
# =============================================================================

def create_depth_anything_v2_pipeline(progress_callback=None):
    """
    Creates Depth Anything V2 pipeline - 10x faster than diffusion-based models
    with better accuracy. Available in Small (25M), Base (98M), Large (335M).
    """
    # Clear GPU memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if progress_callback:
        progress_callback(10, 100)

    # Using Large model for best quality - can switch to Base or Small for speed
    model_id = "depth-anything/Depth-Anything-V2-Large-hf"

    image_processor = AutoImageProcessor.from_pretrained(model_id)

    if progress_callback:
        progress_callback(30, 100)

    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.to(torch_get_device())
    model.eval()

    if progress_callback:
        progress_callback(50, 100)

    return model, image_processor


def run_depth_anything_v2_pipeline(image, model, image_processor, progress_callback=None):
    """
    Runs Depth Anything V2 inference - optimized for speed and quality.
    """
    # Clear GPU cache to free up memory before running inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    original_size = image.size  # (width, height)

    if progress_callback:
        progress_callback(60, 100)

    # Prepare inputs
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(torch_get_device()) for k, v in inputs.items()}

    if progress_callback:
        progress_callback(70, 100)

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    if progress_callback:
        progress_callback(85, 100)

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(original_size[1], original_size[0]),  # (height, width)
        mode="bicubic",
        align_corners=False,
    )

    # Convert to numpy and normalize
    output = prediction.squeeze().cpu().numpy()

    # Normalize to 0-255 range
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)
    formatted = (output * 255).astype("uint8")

    # Invert so that far = black, near = white (matching other models)
    formatted = 255 - formatted

    if progress_callback:
        progress_callback(100, 100)

    # Free GPU memory after inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return formatted


# =============================================================================
# DINOv2 (Legacy)
# =============================================================================

def create_dinov2_pipeline(progress_callback=None):
    # Clear GPU memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/dpt-dinov2-large-nyu"
    )
    model = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-large-nyu")
    model.to(torch_get_device())
    return model, image_processor


def run_dinov2_pipeline(image, model, image_processor, progress_callback=None):
    # Clear GPU cache to free up memory before running inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image = Image.fromarray(image)

    # Use 512px max to reduce memory usage (down from 1024)
    max_size = 512
    new_size = image.size
    if image.width > image.height:
        if image.width > max_size:
            new_size = (max_size, int(image.height * max_size / image.width))
    else:
        if image.height > max_size:
            new_size = (int(image.width * max_size / image.height), max_size)

    resized_image = image.convert("RGB").resize(new_size, Image.BICUBIC)
    inputs = image_processor(images=resized_image, return_tensors="pt")
    inputs = {k: v.to(torch_get_device()) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    formatted[:, :] = 255 - formatted[:, :]  # invert the depth map

    # resize to original size
    formatted = cv2.resize(
        formatted, (image.width, image.height), interpolation=cv2.INTER_CUBIC
    )

    # Free GPU memory after inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return formatted


def create_medias_pipeline(progress_callback=None):
    """
    Creates a media pipeline using the MiDaS model for depth estimation.

    Args:
        progress_callback (callable, optional): A callback function to report progress. Defaults to None.

    Returns:
        tuple: A tuple containing the MiDaS model and the transformation pipeline.

    """
    # Clear GPU memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load the MiDaS v2.1 model
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, skip_validation=True)

    if progress_callback:
        progress_callback(30, 100)

    # Set the model to evaluation mode
    midas.eval()

    # Define the transformation pipeline
    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS", "transforms", skip_validation=True
    )
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transforms = midas_transforms.dpt_transform
    else:
        transforms = midas_transforms.small_transform

    if progress_callback:
        progress_callback(50, 100)

    # Set the device (CPU or GPU)
    midas.to(torch_get_device())

    return midas, transforms


def run_medias_pipeline(image, midas, transforms, progress_callback=None):
    """
    Runs the media pipeline for segmentation.

    Args:
        image (numpy.ndarray): The input image.
        midas (torch.nn.Module): The MIDAS model.
        transforms (torchvision.transforms.Compose): The image transforms.
        progress_callback (callable, optional): A callback function to report progress.

    Returns:
        numpy.ndarray: The predicted segmentation mask.
    """
    # Clear GPU cache to free up memory before running inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    input_batch = transforms(image).to(torch_get_device())
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    if progress_callback:
        progress_callback(90, 100)

    result = prediction.cpu().numpy()

    # Free GPU memory after inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def midas_depth_map(image, progress_callback=None):
    if progress_callback:
        progress_callback(0, 100)

    midas, transforms = create_medias_pipeline(progress_callback=progress_callback)

    depth_map = run_medias_pipeline(
        image, midas, transforms, progress_callback=progress_callback
    )

    if progress_callback:
        progress_callback(100, 100)

    return depth_map


def create_zoedepth_pipeline(progress_callback=None):
    # Clear GPU memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Triggers fresh download of MiDaS repo
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)

    # Zoe_NK
    model_zoe_nk = torch.hub.load(
        "isl-org/ZoeDepth", "ZoeD_NK", pretrained=True, skip_validation=True
    )

    # Set the device (CPU or GPU)
    device = torch_get_device()
    model_zoe_nk.to(device)

    if progress_callback:
        progress_callback(50, 100)

    return model_zoe_nk


def run_zoedepth_pipeline(image, model_zoe_nk, progress_callback=None):
    # Clear GPU cache to free up memory before running inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    depth_map = model_zoe_nk.infer_pil(image)  # as numpy

    # invert the depth map since we are expecting the farthest objects to be black
    depth_map = 255 - depth_map

    if progress_callback:
        progress_callback(100, 100)

    # Free GPU memory after inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return depth_map


def zoedepth_depth_map(image, progress_callback=None):
    model_zoe_nk = create_zoedepth_pipeline(progress_callback=progress_callback)

    return run_zoedepth_pipeline(
        image, model_zoe_nk, progress_callback=progress_callback
    )
