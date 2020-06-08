from skimage.filters import try_all_threshold
from skimage.color import rgb2gray
from skimage.io import imread
from matplotlib import pyplot as plt
import numpy as np

img = imread('maze.jpeg')
img = rgb2gray(img)

fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
plt.show()
