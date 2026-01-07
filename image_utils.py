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
    img = image.astype(np.float32)

    # אם הערכים כבר בטווח 0..1 (float) – לא מחלקים שוב ב-255
    if img.max() > 1.0:
        img = img / 255.0

    if img.ndim == 3:
        gray = rgb2gray(img)
    else:
        gray = img

    edges = sobel(gray)
    edges = np.power(edges, 0.25)

    m = edges.max()
    if m > 0:
        edges = edges / m

    return np.clip(np.rint(edges * 255), 0, 255).astype(np.uint8)
