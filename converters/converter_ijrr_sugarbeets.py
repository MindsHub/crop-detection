from common_converters import *

RGB_PATH = "datasets_raw/ijrr_sugarbeets_2016_annotations/CKA_160523/images/rgb/"
IMAP_PATH = "datasets_raw/ijrr_sugarbeets_2016_annotations/CKA_160523/annotations/dlp/iMapCleaned/"
COLOR_PATH = "datasets_raw/ijrr_sugarbeets_2016_annotations/CKA_160523/annotations/dlp/colorCleaned/"
TYPES = {
	TRAINING: 1000,
	VALIDATION: 250,
}

files = getFiles(RGB_PATH, IMAP_PATH, ignoreUnexistingLabels=True)
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
