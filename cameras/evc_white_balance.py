import numpy as np

def evc_white_balance(input_image: np.ndarray, white: np.ndarray) -> np.ndarray:

    # Check if white is zero to avoid division by zero
    if np.all(white == 0):
        return input_image
    
    # Normalize the image with the white point
    normalized = input_image / white
    
    # Clip the pixel values to the range [0, 1]
    clipped = np.clip(normalized, 0, 1)
    
    # Return the result
    result = np.minimum(clipped, 1)
    return result
