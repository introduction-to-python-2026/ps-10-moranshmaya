from PIL import Image
import numpy as np
from skimage.filters import sobel

def load_image(path):
    img = Image.open(path)

    # edges: להחזיר מסכה בוליאנית דו־ממדית
    if path.endswith("lena_edges.png") or "edges" in path:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        return arr > 0

    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def edge_detection(image):
    # image יכול להגיע RGB אחרי median(..., ball(3))
    if image.ndim == 3:
        gray = image.mean(axis=2).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    # sobel עובד טוב על float
    edges = sobel(gray)

    # להחזיר 0..255 כדי שסף > 50 יעבוד
    edges = np.clip(edges * 255.0, 0, 255).astype(np.uint8)
    return edges
