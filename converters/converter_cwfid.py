import glob
import random
import cv2
import os
import numpy as np

IMAGES_PATH = "datasets_raw/cwfid/images/"
MASKS_PATH = "datasets_raw/cwfid/masks/"
TO = "../dataset/{}/{}/"
TYPES = {
	0: "train",
	1: "test"
}
TYPES_DISTRIBUTION = [0] * 1000 + [1] * 250 + [2] * 10
random.shuffle(TYPES_DISTRIBUTION)

files = []
for imageFile in sorted(glob.glob(IMAGES_PATH + "*.png")):
	_, filename = os.path.split(imageFile)
	maskFile = MASKS_PATH + filename.replace("image", "mask")
	files.append((imageFile, maskFile, int(filename[:3])))

def crop(image, mask, number, x1, y1, x2, y2, transpose):
	image1 = image[x1:x2, y1:y2, :]
	mask1 = mask[x1:x2, y1:y2]

	if transpose:
		image1 = cv2.transpose(image1)
		#image1 = cv2.flip(image1, flipCode=0)
		mask1 = cv2.transpose(mask1)
		#mask1 = cv2.flip(mask1, flipCode=0)

	assert np.shape(image1)[:2] == (352, 480)
	return image1, mask1, f"cwfid_{number:0>3}_cropped_{x1}_{y1}_{x2}_{y2}.png"

def flipBoth(image, mask, number, flipCode):
	return cv2.flip(image, flipCode=flipCode), cv2.flip(mask, flipCode=flipCode), f"cwfid_{number:0>3}_scaled_{flipCode}.png"

def extractImagesMasks(image, mask, number):
	for x1, x2 in [(0, 352), (307, 659), (614, 966)]:
		for y1, y2 in [(0, 480), (408, 888), (816, 1296)]:
			yield crop(image, mask, number, x1, y1, x2, y2, False)

	for x1, x2 in [(0, 480), (486, 966)]:
		for y1, y2 in [(0, 352), (315, 667), (630, 982), (944, 1296)]:
			yield crop(image, mask, number, x1, y1, x2, y2, True)

	imageScaled = cv2.resize(image, (480, 352), interpolation=cv2.INTER_NEAREST)
	maskScaled = cv2.resize(mask, (480, 352), interpolation=cv2.INTER_NEAREST)
	yield imageScaled, maskScaled, f"cwfid_{number:0>3}_scaled_.png"
	yield flipBoth(imageScaled, maskScaled, number, 0)
	yield flipBoth(imageScaled, maskScaled, number, 1)
	yield flipBoth(imageScaled, maskScaled, number, -1)

i = 0
for imageFile, maskFile, number in files:
	print(imageFile, maskFile, number)
	image = cv2.imread(imageFile, cv2.IMREAD_UNCHANGED)
	mask = cv2.imread(maskFile, cv2.IMREAD_UNCHANGED)
	mask = np.array(mask == 0, dtype=np.uint8)

	for extractedImage, extractedMask, filename in extractImagesMasks(image, mask, number):
		type = TYPES_DISTRIBUTION[i]
		print(i, type, filename)
		i += 1
		if type == 2:
			continue
		type = TYPES[type]

		cv2.imwrite(TO.format(type, "images") + filename, extractedImage)
		cv2.imwrite(TO.format(type, "labels") + filename, extractedMask)

