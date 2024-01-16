import cv2
import numpy as np


def hsv_to_rgb_opencv(hsv_array):
    # Ensure that the input array has the correct shape (N, 3)
    assert hsv_array.shape[1] == 3, "Input array must have shape (N, 3)"
    # Convert HSV array to RGB array using OpenCV
    hsv_array_uint8 = (hsv_array * 255).astype(np.uint8)  # Convert to uint8
    rgb_array_uint8 = cv2.cvtColor(hsv_array_uint8[None], cv2.COLOR_HSV2RGB)[0]
    # Normalize back to [0, 1]
    rgb_array = rgb_array_uint8 / 255.0
    return rgb_array


# Map the 0-1 value to the Hue component in HSV
def generate_gradient_color(value: np.ndarray,
                            start=0.4,
                            end=0.65):
    N = value.shape[0]
    value = (value - value.min()) / (value.max() - value.min())
    value = start + (end - start) * value
    hsv = np.ones((N, 3), dtype=np.float32)
    hsv[:, 0] *= value    
    rgb = hsv_to_rgb_opencv(hsv)

    return rgb


def generate_synthetic_point_cloud(num_points):
    return np.random.rand(num_points, 3) * 2 - 1
