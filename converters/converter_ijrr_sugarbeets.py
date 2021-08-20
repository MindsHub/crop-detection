import glob
import os
import shutil
import cv2

RGB_PATH = "datasets_raw/ijrr_sugarbeets_2016_annotations/CKA_160523/images/rgb/"
IMAP_PATH = "datasets_raw/ijrr_sugarbeets_2016_annotations/CKA_160523/annotations/dlp/iMapCleaned/"
COLOR_PATH = "datasets_raw/ijrr_sugarbeets_2016_annotations/CKA_160523/annotations/dlp/colorCleaned/"
TO = "../dataset/{}/{}/"
TYPES = {
	"train": 1000,
	"test": 250,
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
for type, count in TYPES.items():
	for j in range(count):
		print(j)
		rgbFile, imapFile, filename = files[i+j]
		rgb = cv2.imread(rgbFile, cv2.IMREAD_UNCHANGED)
		rgb = resizeImage(rgb)
		cv2.imwrite(TO.format(type, "images") + filename, rgb)

		imap = cv2.imread(imapFile, cv2.IMREAD_UNCHANGED)
		imap = resizeImage(imap)
		imap[imap > 0] = 1
		cv2.imwrite(TO.format(type, "labels") + filename, imap)
	i += count
