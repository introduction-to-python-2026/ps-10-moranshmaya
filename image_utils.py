from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    def load_image(path):
    # Load image, convert to grayscale, return as float array in [0, 1]
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def edge_detection(image):
    def edge_detection(image):
    # 1) Smooth to reduce noise
    blurred = convolve2d(image, np.ones((3, 3), dtype=np.float32) / 9.0, mode="same", boundary="symm")

    # 2) Sobel filters (detect intensity changes)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)

    gx = convolve2d(blurred, sobel_x, mode="same", boundary="symm")
    gy = convolve2d(blurred, sobel_y, mode="same", boundary="symm")
    # 3) Edge magnitude
    edges = np.sqrt(gx*2 + gy*2)

    # Optional: normalize to [0, 1] for nice output / stability
    max_val = edges.max() if edges.size else 0.0
    if max_val > 0:
        edges = edges / max_val

    return edges
