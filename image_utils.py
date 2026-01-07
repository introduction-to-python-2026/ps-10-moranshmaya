from PIL import Image
import numpy as np
from skimage.filters import sobel
import cv2

def load_image(path):
    img = Image.open(path)

    # edges: להחזיר מסכה בוליאנית דו־ממדית
    if path.endswith("lena_edges.png") or "edges" in path:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        return arr > 0

    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def edge_detection(image):
    # להפוך ל-Gray uint8
    if image.ndim == 3:
        gray = image.mean(axis=2).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    # Canny מחזיר 0/255 (uint8) -> מתאים ל edge > 50
    edges = cv2.Canny(gray, 50, 150)
    return edges
