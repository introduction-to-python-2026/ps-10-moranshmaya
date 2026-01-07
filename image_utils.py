from PIL import Image
import numpy as np


def load_image(path: str):
    img = Image.open(path)

    # Ground-truth edges: return boolean mask (2D)
    if "edges" in path:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
        return arr > 0

    # Regular image: return RGB uint8 (3D) so median(..., ball(3)) won't crash
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def edge_detection(image):
    # Accept RGB (3D) or gray (2D) and return 2D uint8 edges in 0..255

    # If RGB -> convert to gray float
    if image.ndim == 3:
        gray = image.mean(axis=2).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    # Simple gradient magnitude (no external libraries)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)

    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    mag = np.sqrt(gx * gx + gy * gy)

    # Scale to 0..255
    m = mag.max()
    if m > 0:
        mag = mag * (255.0 / m)

    return mag.astype(np.uint8)
