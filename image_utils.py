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
    # to float32
    img = image.astype(np.float32)

    # normalize to 0..1 (works for uint8 and for float 0..255)
    if img.max() > 1.0:
        img = img / 255.0

    # grayscale in 0..1
    gray = rgb2gray(img) if img.ndim == 3 else img

    # sobel response
    edges = sobel(gray)

    # normalize to 0..1
    m = edges.max()
    if m > 0:
        edges = edges / m

    # to uint8 0..255 (עם עיגול, זה חשוב לסף 50)
    return np.rint(edges * 255).astype(np.uint8)
