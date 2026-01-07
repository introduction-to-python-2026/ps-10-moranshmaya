from PIL import Image
import numpy as np
from skimage.filters import sobel


def load_image(path):
    img = Image.open(path)

    # לתמונת קצוות – להחזיר בוליאני
    if "edges" in path:
        return np.asarray(img.convert("L")) > 0

    # לתמונה רגילה – להחזיר RGB (3 ממדים)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def edge_detection(image):
    # להפוך RGB לאפור
    if image.ndim == 3:
        image = image.mean(axis=2)

    img = image.astype(np.int16)

    dx = np.abs(img[:, 1:] - img[:, :-1])
    dy = np.abs(img[1:, :] - img[:-1, :])

    dx = np.pad(dx, ((0, 0), (0, 1)))
    dy = np.pad(dy, ((0, 1), (0, 0)))

    edges = dx + dy
    edges = np.clip(edges, 0, 255).astype(np.uint8)

    return edges
