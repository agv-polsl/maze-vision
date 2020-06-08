import math
from skimage import io
from skimage.filters import threshold_isodata
from skimage.color import rgb2gray, rgb2hsv
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion
import numpy as np


def unit_circle(r):
    d = 2 * r + 1
    rx, ry = d/2, d/2
    x, y = np.indices((d, d))
    return (np.abs(np.hypot(rx - x, ry - y)-r) < 0.5).astype(int)


img = io.imread('maze.jpg')
img_hsv = rgb2hsv(img)
img_gray = rgb2gray(img)

# Find red markers
# Red hue is near 0.0
red_dots_bin = np.logical_or(img_hsv[:, :, 0] < 0.1, img_hsv[:, :, 0] > 0.9)
# Moderate saturation treshold
red_dots_bin &= img_hsv[:, :, 1] > 0.3
# Moderate value treshold
red_dots_bin &= img_hsv[:, :, 2] > 0.3
red_dots_bin = red_dots_bin.astype(np.uint8)

markers_labels = label(red_dots_bin)
markers_regions = regionprops(markers_labels)
markers = []
diams = []
for props in markers_regions:
    markers.append(props.centroid)
    diams.append(props.equivalent_diameter)
max_marker_diam = math.ceil(max(diams))

plt.figure()
plt.imshow(red_dots_bin, cmap='gray')
for marker in markers:
    plt.plot(marker[1], marker[0], '.r')
plt.title('Markers')

# Walls binarisation
thr = threshold_isodata(img_gray)
img_bin = img_gray > thr
img_bin = img_bin.astype(np.uint8)
plt.figure()
plt.imshow(img_bin, cmap='gray')
plt.title('Binarised')

# Remove marker dots
dilated = binary_dilation(red_dots_bin,
                          np.ones((max_marker_diam * 2,
                                   max_marker_diam * 2)))
img_bin = np.clip(img_bin + dilated, 0, 1)

plt.figure()
plt.imshow(img_bin, cmap='gray')
plt.title('Binarised with markers removed')

# Expand walls
diam = np.linalg.norm(np.array(markers[0]) - np.array(markers[1]))
margin = int(0.55 * diam)

# Dilate markers remainings
#img_bin_exp = binary_dilation(img_bin)
img_bin_exp = binary_erosion(img_bin, unit_circle(margin))

plt.figure()
plt.imshow(img_bin_exp, cmap='gray')
# diameter_opening

plt.show()
