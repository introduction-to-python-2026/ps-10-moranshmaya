from PIL import Image
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
def load_image(path: str):
    img = Image.open(path)
    if "edges" in path:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        return arr > 0
    return np.asarray(img.convert("RGB"), dtype=np.uint8)
def edge_detection(image):
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    gray = rgb2gray(img) if img.ndim == 3 else img
    edges = sobel(gray)
    return np.clip(edges * 255.0, 0, 255).astype(np.uint8)
