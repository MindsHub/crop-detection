import os
import glob
import cv2
import numpy as np

if not os.path.exists("./.git/") or not os.path.exists("./crop_detection/"):
	print("Run this script from the root of the crop-detection repo!")
	exit(1)

DATASET_PATH = "dataset/{}/{}/"
TRAINING = "training"
VALIDATION = "validation"
IMAGES = "images"
LABELS = "labels"
WIDTH = 480
HEIGHT = 352

for typ1 in [TRAINING, VALIDATION]:
	for typ2 in [IMAGES, LABELS]:
		os.makedirs(DATASET_PATH.format(typ1, typ2), exist_ok=True)


def resizeImage(image):
	return cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

def crop(image, label, number, x, y, transpose):
	x1, x2 = x, x + (WIDTH if transpose else HEIGHT)
	y1, y2 = y, y + (HEIGHT if transpose else WIDTH)
	image1 = image[x1:x2, y1:y2, :]
	label1 = label[x1:x2, y1:y2]

	if transpose:
		image1 = cv2.transpose(image1)
		#image1 = cv2.flip(image1, flipCode=0)
		label1 = cv2.transpose(label1)
		#label1 = cv2.flip(label1, flipCode=0)

	if np.shape(image1)[:2] != (HEIGHT, WIDTH):
		print(x1, x2, y1, y2, transpose, np.shape(image1)[:2], (HEIGHT, WIDTH))
		assert False
	return image1, label1, f"{number:0>3}_cropped_{x1}_{y1}_{x2}_{y2}.png"

def flipBoth(image, label, number, flipCode):
	return cv2.flip(image, flipCode=flipCode), cv2.flip(label, flipCode=flipCode), f"{number:0>3}_scaled_{flipCode}.png"

def getFiles(imagesPath, labelsPath, editFilename=lambda x: x, ignoreUnexistingLabels=False):
	files = []
	for imageFile in sorted(glob.glob(os.path.join(imagesPath, "*.png"))):
		_, filename = os.path.split(imageFile)
		filename = editFilename(filename)
		labelFile = os.path.join(labelsPath, filename)
		if os.path.exists(labelFile):
			files.append((imageFile, labelFile, filename))
		elif not ignoreUnexistingLabels:
			print(f"label file does not exist: {labelFile}")
			exit(1)
	files = sorted(files, key=lambda x: x[2])
	return files