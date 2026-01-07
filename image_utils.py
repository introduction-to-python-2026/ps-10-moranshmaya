from PIL import Image
import numpy as np

def load_image(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    return arr

def edge_detection(image):
    # Simple edge detection using differences (no scipy)

    # Horizontal edges
    dx = np.abs(image[:, 1:] - image[:, :-1])

    # Vertical edges
    dy = np.abs(image[1:, :] - image[:-1, :])

    # Pad to original size
    dx = np.pad(dx, ((0, 0), (0, 1)))
    dy = np.pad(dy, ((0, 1), (0, 0)))

    edges = dx + dy

    # Normalize to [0, 1]
    max_val = edges.max()
    if max_val > 0:
        edges = edges / max_val

    return edges
