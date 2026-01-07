import numpy as np
from PIL import Image


def load_image(path):
    img = Image.open(path).convert("L")
    return np.array(img)


def edge_detection(img):
    # Ensure float computations
    img = img.astype(np.float32)

    # Pad to handle borders
    p = np.pad(img, 1, mode="edge")

    # Sobel gradients (3x3) implemented with pure numpy shifts
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])

    mag = np.hypot(gx, gy)

    # Normalize to 0..255
    mmax = mag.max()
    if mmax > 0:
        mag = (mag / mmax) * 255.0

    return mag.astype(np.uint8)
