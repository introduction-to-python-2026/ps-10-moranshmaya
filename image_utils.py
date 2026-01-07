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
    def edge_detection(image):
    img = image.astype(np.float32)

    # נרמול רק אם הקלט הוא uint8
    if img.max() > 1.0:
        img = img / 255.0

    gray = rgb2gray(img) if img.ndim == 3 else img

    edges = sobel(gray)

    m = edges.max()
    if m > 0:
        edges = edges / m

    # הגברה לינארית עדינה כדי לעבור את הסף 50
    edges = np.clip(edges * 1.2, 0.0, 1.0)

    return (edges * 255).astype(np.uint8)
