import random
from common_converters import *

IMAGES_PATH = "datasets_raw/cwfid/images/"
LABELS_PATH = "datasets_raw/cwfid/masks/"

TYPES = {
	0: TRAIN,
	1: TEST,
	# the 2 is skipped
}
TYPES_DISTRIBUTION = [0] * 1000 + [1] * 250 + [2] * 10
random.Random(4).shuffle(TYPES_DISTRIBUTION)

files = getFiles(IMAGES_PATH, LABELS_PATH, editFilename=lambda filename: filename.replace("image", "mask"))

def extractImagesLabels(image, label, number):
	for x in [0, 307, 614]: # they overlap a bit
		for y in [0, 408, 816]:
			yield crop(image, label, number, x, y, False)

	for x in [0, 486]:
		for y in [0, 315, 630, 944]:
			yield crop(image, label, number, x, y, True)

	imageScaled = resizeImage(image)
	labelScaled = resizeImage(label)
	yield imageScaled, labelScaled, f"{number:0>3}_scaled_.png"
	yield flipBoth(imageScaled, labelScaled, number, 0)
	yield flipBoth(imageScaled, labelScaled, number, 1)
	yield flipBoth(imageScaled, labelScaled, number, -1)

i = 0
for imageFile, labelFile, filename in files:
	print(imageFile, labelFile, filename)
	image = cv2.imread(imageFile, cv2.IMREAD_UNCHANGED)
	label = cv2.imread(labelFile, cv2.IMREAD_UNCHANGED)
	label = np.array(label == 0, dtype=np.uint8)
	number = int(filename[:3])

	for extractedImage, extractedLabel, filename in extractImagesLabels(image, label, number):
		typ = TYPES_DISTRIBUTION[i]
		print(f"{i: 4} / {len(TYPES_DISTRIBUTION)} - {TYPES[typ] if typ in TYPES else '': >5} - {filename}")
		i += 1
		if typ == 2:
			continue
		typ = TYPES[typ]

		cv2.imwrite(DATASET_PATH.format(typ, IMAGES) + "cwfid_" + filename, extractedImage)
		cv2.imwrite(DATASET_PATH.format(typ, LABELS) + "cwfid_" + filename, extractedLabel)
