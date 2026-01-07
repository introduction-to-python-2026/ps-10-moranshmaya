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
    pass # Replace the `pass` with your code
