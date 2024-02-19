import numpy as np

def evc_compute_binary(input_image: np.ndarray, x: float, top: int) -> np.ndarray:

    # compute a binary image with the threshold x
    binary_image = np.zeros_like(input_image)
    binary_image[input_image > x] = 1

    # invert the binary image if top is 0
    if top == 0:
        binary_image = 1 - binary_image

    return binary_image.astype(float)
