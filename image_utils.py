from PIL import Image
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel


def load_image(path: str):
    img = Image.open(path)

    # ground-truth edge mask (2D boolean)
    if "edges" in path:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        return arr > 0

    # regular image (RGB uint8, 3D)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def edge_detection(image):
    # RGB -> gray in skimage convention (float 0..1)
    if image.ndim == 3:
        gray = rgb2gray(image)
    else:
        gray = image.astype(np.float32) / 255.0

    edges = sobel(gray)
    edges = (edges * 255).astype(np.uint8)
    return edges
