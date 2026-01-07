from PIL import Image
import numpy as np
from skimage.filters import sobel


def load_image(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img)

    if "edges" in path:
        return arr > 0

    return arr.astype(np.uint8)


def edge_detection(image):
    # sobel מחזיר ערכים בין 0 ל־1
    edges = sobel(image)

    # להחזיר בסקאלה של 0..255 כדי ש־edge > 50 יעבוד
    edges = (edges * 255).astype(np.uint8)

    return edges
