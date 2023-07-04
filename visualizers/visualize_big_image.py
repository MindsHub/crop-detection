import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path = ['./'] + sys.path
from crop_detection import segmentImage

import glob
import cv2
import numpy as np

IMAGES = (
    glob.glob("./datasets_raw/cyberorto/original/*") +
    glob.glob("./visualizers/test_images/*")
)
CHECKPOINT_PATH = "./checkpoint_352x480_dataset3"
EPOCH = 30

modelPath = f"{CHECKPOINT_PATH}/model_{EPOCH}.hd5"
imageBasePath = f"{CHECKPOINT_PATH}/bigImage{EPOCH}"

for imagePath in IMAGES:
    folders, filename = os.path.split(imagePath)
    outputPath = "_".join([
        imageBasePath,
        folders.replace(".", "").replace("/", "_"),
        filename.replace(".png", ".jpg")
    ])
    print(outputPath)

    image = cv2.imread(imagePath)
    segmented = segmentImage(image, modelPath=modelPath)
    out2 = np.array((segmented > 128) * 255, dtype=np.uint8)
    out3 = out2

    erodeDilateKernel = np.ones((3, 3), dtype=np.uint8)
    out3 = cv2.dilate(out3, erodeDilateKernel, iterations=3)
    out3 = cv2.erode(out3, erodeDilateKernel, iterations=3)

    erodeDilateKernel = np.ones((2, 2), dtype=np.uint8)
    out3 = cv2.erode(out3, erodeDilateKernel, iterations=3)
    out3 = cv2.dilate(out3, erodeDilateKernel, iterations=3)

    overlay = cv2.addWeighted(cv2.cvtColor(out3, cv2.COLOR_GRAY2RGB), 0.4, image, 0.6, 0)

    h, w, c = image.shape
    result = np.empty((h*2, w*2, c))
    result[:h, :w, :] = image
    result[:h, w:, :] = cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)
    result[h:, w:, :] = cv2.cvtColor(out3, cv2.COLOR_GRAY2RGB)
    result[h:, :w, :] = overlay

    if not cv2.imwrite(outputPath, result):
        print("Unable to save image to", outputPath)
        break
