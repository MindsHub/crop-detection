import sys
import os
import cv2
import numpy as np

folder, filename = os.path.split(sys.argv[0])
filename = filename[:-2] + "png"

def getFilename(subfolder):
    os.makedirs(os.path.join(folder, "..", subfolder), exist_ok=True)
    return os.path.join(folder, "..", subfolder, filename)

def exportImageAndLabel(image, label):
    cv2.imwrite(getFilename("images"), image)
    cv2.imwrite(getFilename("labels"), label * 255)

image = cv2.imread(getFilename("original"), cv2.IMREAD_UNCHANGED)

# remove transparent parts
image = image[np.r_[283:1891, 2159:3767], :, :3]

# cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
# cv2.imshow("image", image)
# while cv2.waitKey(1) & 0xFF != ord('q'): pass