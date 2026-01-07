from PIL import Image
import numpy as np

def load_image(path):
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)

    if "edges" in path:
        return arr > 0

    return arr
    

def edge_detection(image):
   def edge_detection(image):
    img = image.astype(np.int16)

    dx = np.abs(img[:, 1:] - img[:, :-1])
    dy = np.abs(img[1:, :] - img[:-1, :])

    dx = np.pad(dx, ((0, 0), (0, 1)))
    dy = np.pad(dy, ((0, 1), (0, 0)))

    edges = dx + dy
    edges = np.clip(edges, 0, 255).astype(np.uint8)

    return edges
  
