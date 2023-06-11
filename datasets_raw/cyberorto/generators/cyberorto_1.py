import cv2
import numpy as np
from read_original_image import *

# the rest has some different colors that wouldn't be properly segmented
image = image[:,402:4690]

segmented = np.zeros(image.shape[0:2])
segmented -= image[:,:,0] / 2
segmented += image[:,:,1]
segmented -= image[:,:,2] / 2
segmented = (segmented[:,:] > -10)

nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(np.asarray(segmented, np.uint8))
sizes = stats[:, -1][1:]
nb_blobs -= 1
min_size = 100

withoutSmallConnectedComponents = np.zeros((segmented.shape), bool)
x = 0
for blob in range(nb_blobs):
    if sizes[blob] >= min_size:
        # see description of im_with_separated_blobs above
        withoutSmallConnectedComponents[im_with_separated_blobs == blob + 1] = True
        x += 1
        if x % 10 == 0:
            print(x, end=" ", flush=True)
print(x)

exportImageAndLabel(image, withoutSmallConnectedComponents)