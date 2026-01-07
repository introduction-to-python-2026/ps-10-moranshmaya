from PIL import Image
import numpy as np
from scipy.signal import convolve2d
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
    if img.ndim == 3:
        gray = img.mean(axis=2)
    else:
        gray = img
    kernelY = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]], dtype=np.float32)
    kernelX = np.array([[ 1,  0, -1],
                        [ 2,  0, -2],
                        [ 1,  0, -1]], dtype=np.float32)
    edgeY = convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)
    edgeX = convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    edgeMAG = np.nan_to_num(edgeMAG, nan=0.0, posinf=0.0, neginf=0.0)
    m = edgeMAG.max()
    if m > 0:
        edgeMAG = edgeMAG / m
  
return np.clip(edgeMAG * 255.0, 0, 255).astype(np.uint8)
