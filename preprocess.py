"""
Image Preprocessing for Neural Network Photo Recognition.
Uses ONLY Pillow for opening, converting to grayscale, and resizing.
Output shape: (features, examples) = (1024, m).
Label encoding: injured = 1, healthy = 0.
"""

import io
from typing import Union, BinaryIO
import numpy as np
from PIL import Image

IMAGE_SIZE = (32, 32)
NUM_FEATURES = 32 * 32  # 1024


def preprocess_image(image_input: Union[str, bytes, BinaryIO, Image.Image]) -> np.ndarray:
    """
    Loads an image, converts to grayscale ('L'), resizes to 32x32,
    normalizes pixel values to [0, 1], and flattens to a (1024, 1) column vector.
    
    Args:
        image_input: File path, raw bytes, file-like object, or PIL Image.
        
    Returns:
        np.ndarray of shape (1024, 1) and dtype float64.
    """
    if isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        # File-like object
        img = Image.open(image_input)

    # 1. Convert to grayscale ('L')
    img_gray = img.convert("L")

    # 2. Resize to 32x32 using Bilinear resampling
    img_resized = img_gray.resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)

    # 3. Convert to numpy array, normalize to [0, 1]
    arr = np.asarray(img_resized, dtype=np.float64) / 255.0

    # 4. Flatten to (1024, 1) column vector (features, 1)
    vec = arr.reshape(-1, 1)
    assert vec.shape == (NUM_FEATURES, 1), f"Expected shape ({NUM_FEATURES}, 1), got {vec.shape}"

    return vec


def flip_image_vector(vec: np.ndarray) -> np.ndarray:
    """
    Horizontally flips a flattened 1024-dimension image vector.
    
    Args:
        vec: Column vector of shape (1024, 1) or (1024,).
        
    Returns:
        Flipped column vector of shape (1024, 1).
    """
    arr = vec.reshape(IMAGE_SIZE)
    flipped = np.fliplr(arr)
    return flipped.reshape(-1, 1)
