import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path = ['./'] + sys.path
from crop_detection import segmentImage

import glob
import cv2

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
    segmentedPath = "_".join([
        imageBasePath,
        folders.replace(".", "").replace("/", "_"),
        filename.replace(".jpg", ".png")
    ])
    print(segmentedPath)

    segmented = segmentImage(imagePath, modelPath=modelPath)
    if not cv2.imwrite(segmentedPath, segmented):
        print("Unable to save image to", segmentedPath)
        break
