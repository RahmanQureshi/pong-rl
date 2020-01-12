import numpy as np

def rgb_to_grayscale(ob):
    """Convert rgb to grayscale. Uses observed luminance to compute grey value.
    """
    rgb = np.array([0.299, 0.587, 0.114])
    return np.inner(rgb, ob).astype(np.uint8)