from common_converters import *

RGB_PATH = "datasets_raw/synthetic_sugarbeat_random_weeds/rgb/"
GT_PATH = "datasets_raw/synthetic_sugarbeat_random_weeds/gt/"
TYPES = {
	TRAINING: 500,
	VALIDATION: 125,
}

files = getFiles(RGB_PATH, GT_PATH)

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
