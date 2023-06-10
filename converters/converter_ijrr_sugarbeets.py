import glob
import os
import shutil
import cv2
from setup_dataset_path import *

RGB_PATH = "datasets_raw/ijrr_sugarbeets_2016_annotations/CKA_160523/images/rgb/"
IMAP_PATH = "datasets_raw/ijrr_sugarbeets_2016_annotations/CKA_160523/annotations/dlp/iMapCleaned/"
COLOR_PATH = "datasets_raw/ijrr_sugarbeets_2016_annotations/CKA_160523/annotations/dlp/colorCleaned/"
TYPES = {
	TRAIN: 1000,
	TEST: 250,
}

def getRanges(arr):
	res = []
	begin = arr[0]
	last = arr[0]
	for e in arr[1:] + [1e100]:
		if e != last + 1:
			res.append(str(begin) if begin == last else f"{begin}-{last}")
			begin = e
		last = e
	return res

def resizeImage(image):
	return cv2.resize(image, (480, 352), interpolation=cv2.INTER_NEAREST)


files = []
for rgbFile in sorted(glob.glob(RGB_PATH + "*.png")):
	_, filename = os.path.split(rgbFile)
	imapFile = IMAP_PATH + filename
	if os.path.exists(imapFile):
		files.append((rgbFile, imapFile, filename))

print(len(files))

i = 0
for typ, count in TYPES.items():
	for j in range(count):
		rgbFile, imapFile, filename = files[i+j]
		print(f"{i+j: 4} / {sum(TYPES.values())} - {typ} - {filename}")

		rgb = cv2.imread(rgbFile, cv2.IMREAD_UNCHANGED)
		rgb = resizeImage(rgb)
		cv2.imwrite(DATASET_PATH.format(typ, IMAGES) + filename, rgb)

		imap = cv2.imread(imapFile, cv2.IMREAD_UNCHANGED)
		imap = resizeImage(imap)
		imap[imap > 0] = 1
		cv2.imwrite(DATASET_PATH.format(typ, LABELS) + filename, imap)
	i += count
