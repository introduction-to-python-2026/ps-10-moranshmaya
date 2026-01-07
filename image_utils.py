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
    if image.ndim == 3:
        gray = rgb2gray(image.astype(np.float32) / 255.0)
    else:
        gray = image.astype(np.float32) / 255.0

    edges = sobel(gray)
    m = edges.max()
    if m > 0:
        edges = edges / m 
    return np.clip(np.rint(edges * 255) ,0 ,255).astype(np.uint8)
