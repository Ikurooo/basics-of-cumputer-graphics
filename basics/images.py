import numpy as np
import scipy.ndimage
from PIL import Image

import utils

def read_img(inp: str) -> Image.Image:
    """
    Returns a PIL Image given by its input path.
    """
    img = Image.open(inp)
    return img

def convert(img: Image.Image) -> np.ndarray:
    """
    Converts a PIL image [0, 255] to a numpy array [0, 1].
    """
    return np.array(img, dtype=np.float32) / 255.0

def switch_channels(img: np.ndarray) -> np.ndarray:
    """
    Swaps the red and green channel of an RGB image given by a numpy array.
    """
    return img[:, :, [1, 0, 2]]

def image_mark_green(img: np.ndarray) -> np.ndarray:
    """
    Returns a numpy-array (HxW) with 1 where the green channel of the input image is greater or equal than 0.7, otherwise zero.
    """
    return (img[:, :, 1] >= 0.7).astype(int)

def image_masked(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Sets the pixels of the input image to zero where the mask is 1.
    """
    return img * (1 - mask[:, :, np.newaxis])

def grayscale(img: np.ndarray) -> np.ndarray:
    """
    Returns a grayscale image of the input. Use utils.rgb2gray().
    """
    return utils.rgb2gray(img)

def cut_and_reshape(img_gray: np.ndarray) -> np.ndarray:
    """
    Cuts the image in half (x-dim) and stacks it together in y-dim.
    """
    height, width = img_gray.shape
    half_width = width // 2
    reshaped_image = np.vstack([img_gray[:, :half_width], img_gray[:, half_width:]])
    return reshaped_image

def filter_image(img: np.ndarray) -> np.ndarray:
    """
    Applies a Gaussian filter to the input image using convolution.
    """
    size = 5
    sigma = 2.0
    channels = img.shape[2] if len(img.shape) == 3 else 1

    # Generate 2D Gaussian filter for a single channel
    single_channel_gaussian = utils.gauss_filter(size, sigma)

    # Stack the 2D filter for each channel
    gaussian_kernel = np.stack([single_channel_gaussian] * channels, axis=-1)

    filtered_image = scipy.ndimage.convolve(img, gaussian_kernel, mode='constant', cval=0.0)

    # Apply additional post-processing if needed
    # Example: Increase contrast
    filtered_image = np.clip(filtered_image * 0.3, 0, 1)

    return filtered_image


def horizontal_edges(img: np.ndarray) -> np.ndarray:
    """
    Defines a Sobel kernel to extract horizontal edges and convolves the image with it.
    """
    # Convert RGB image to grayscale
    img_gray = grayscale(img)

    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

    # Use scipy.ndimage.convolve for convolution
    edges = scipy.ndimage.convolve(img_gray, kernel, mode='constant', cval=0.0)

    return edges
