import numpy as np
from PIL import Image
from skimage.filters import sobel


def load_image(path):
    img = Image.open(path).convert("L")
    return np.array(img)


def edge_detection(img):
    edges = sobel(img)
    return (edges * 255).astype(np.uint8)
