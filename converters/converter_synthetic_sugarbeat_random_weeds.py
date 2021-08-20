import glob
import shutil
import cv2
import os

FROM = "datasets_raw/synthetic_sugarbeat_random_weeds/gt/B_*.png"
TO = "../training/dataset/{}/labels/"
TYPES = {
	"train": 500,
	"test": 125,
}

imageCount = 0
for _, count in TYPES.items():
	imageCount += count

files = glob.glob(FROM)
if len(files) != imageCount:
	print(f"{len(files)} files were found but the image count should be {imageCount}")
	exit(1)
files = sorted(files)

i = 0
for type, count in TYPES.items():
	to = TO.format(type)
	for j in range(count):
		image = cv2.imread(files[i+j], cv2.IMREAD_UNCHANGED)
		# TODO resize to 480x352
		image[image > 0] = 1
		filename = to + os.path.split(files[i+j])[1]
		cv2.imwrite(filename, image)
	i += count
