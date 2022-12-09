import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from utils import my_imread
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage import morphology

#img = img_as_float(astronaut()[::2, ::2]) # ndarray: H,W,C
img_dir = 'data/examples/img_rgb.jpg'
img = my_imread(img_dir)
#img = Image.fromarray(img_arr,'RGB')

## felzenszwalb
segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
# slic
segments_slic = slic(img, n_segments=250, compactness=10, sigma=1,
                     start_label=1)
# maskslic
# Compute a mask
lum = rgb2gray(img)
mask = morphology.remove_small_holes(
    morphology.remove_small_objects(
        lum < 0.7, 500),
    500)
segments_mslic = slic(img, n_segments=1000, mask=mask, start_label=1)

# quickshift
segments_quick = quickshift(img, kernel_size=5, max_dist=19, ratio=0.5)
# watershed
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=250, compactness=0.001)

print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')
print(f'SLIC number of segments: {len(np.unique(segments_slic))}')
print(f'MaskSLIC number of segments: {len(np.unique(segments_mslic))}')
print(f'Quickshift number of segments: {len(np.unique(segments_quick))}')
print(f'Watershed number of segments: {len(np.unique(segments_watershed))}')

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_mslic))
ax[0, 0].set_title("MaskSLIC")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()