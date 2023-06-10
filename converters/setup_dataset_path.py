import os

if not os.path.exists("./.git/") or not os.path.exists("./crop_detection/"):
	print("Run this script from the root of the crop-detection repo!")
	exit(1)

DATASET_PATH = "dataset/{}/{}/"
TRAIN = "train"
TEST = "test"
IMAGES = "images"
LABELS = "labels"

for typ1 in [TRAIN, TEST]:
	for typ2 in [IMAGES, LABELS]:
		os.makedirs(DATASET_PATH.format(typ1, typ2), exist_ok=True)