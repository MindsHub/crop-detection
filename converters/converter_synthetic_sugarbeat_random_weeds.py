import glob
import shutil
import cv2
import os
from setup_dataset_path import *

RGB_PATH = "datasets_raw/synthetic_sugarbeat_random_weeds/rgb/"
GT_PATH = "datasets_raw/synthetic_sugarbeat_random_weeds/gt/"
TYPES = {
	TRAIN: 500,
	TEST: 125,
}

files = []
for rgbFile in sorted(glob.glob(RGB_PATH + "*.png")):
	_, filename = os.path.split(rgbFile)
	gtFile = GT_PATH + filename
	if os.path.exists(gtFile):
		files.append((rgbFile, gtFile, filename))
	else:
		print(f"gt file does not exist: {gtFile}")
		exit(1)

def resizeImage(image):
	return cv2.resize(image, (480, 352), interpolation=cv2.INTER_NEAREST)

i = 0
for typ, count in TYPES.items():
	for j in range(count):
		rgbFile, gtFile, filename = files[i+j]
		print(f"{i+j: 4} / {sum(TYPES.values())} - {typ} - {filename}")

		rgb = cv2.imread(rgbFile, cv2.IMREAD_UNCHANGED)
		rgb = resizeImage(rgb)
		cv2.imwrite(DATASET_PATH.format(typ, IMAGES) + filename, rgb)

		gt = cv2.imread(gtFile, cv2.IMREAD_UNCHANGED)
		gt = resizeImage(gt)
		gt[gt > 0] = 1
		cv2.imwrite(DATASET_PATH.format(typ, LABELS) + filename, gt)
	i += count
