import importlib
import numpy as np
import image_utils
importlib.reload(image_utils)

from image_utils import load_image

image_path = "/content/Sandy.jpg"
loaded_image = load_image(image_path)

import matplotlib.pyplot as plt
from image_utils import load_image, edge_detection

from skimage.filters import median
from skimage.morphology import ball

clean_image = median(loaded_image, ball(1))

clean_image = edge_detection(clean_image)
clean_image = np.rot90(clean_image, k=-1)

plt.figure(figsize=(5, 5))
plt.imshow(clean_image, cmap="gray")
plt.axis("off")
plt.show()

plt.figure(figsize=(3, 3))
plt.hist(clean_image.flatten(), bins=256)

binary_clean_image = np.where(clean_image < 50, 0, 1)
binary_clean_image = np.rot90(binary_clean_image, k=0)

plt.figure(figsize=(5, 5))
plt.imshow(binary_clean_image, cmap="gray")

from PIL import Image

edge_image = Image.fromarray(edge_binary)
edge_image.save("my_edges.png")

from PIL import Image
import matplotlib.pyplot as plt

loaded_edge_image = Image.open("my_edges.png")

loaded_edge_image = np.rot90(loaded_edge_image, k=-1)

plt.figure(figsize=(10, 10))
plt.imshow(loaded_edge_image, cmap="gray")
plt.axis("off")
plt.show()
