import random
from common_converters import *

IMAGES_PATH = "datasets_raw/cyberorto/images/"
LABELS_PATH = "datasets_raw/cyberorto/labels/"

TYPES = {
    0: TRAIN,
    1: TEST,
    # the 2 is skipped
}
# currently there is only one image, from which only 144 parts can be taken
TYPES_DISTRIBUTION = [0] * 112 + [1] * 28 + [2] * 4
random.Random(4).shuffle(TYPES_DISTRIBUTION)

files = getFiles(IMAGES_PATH, LABELS_PATH)

def getNonOverlappingRanges(size, wantedSize):
    num = size // wantedSize
    wantedSize = size // num
    print(num, size, wantedSize)
    return range(0, size-wantedSize+1, wantedSize)

def extractImagesLabels(image, label, number):
    for x in getNonOverlappingRanges(image.shape[0], HEIGHT):
        for y in getNonOverlappingRanges(image.shape[1], WIDTH):
            yield crop(image, label, number, x, y, False)

    for x in getNonOverlappingRanges(image.shape[0], WIDTH):
        for y in getNonOverlappingRanges(image.shape[1], HEIGHT):
            yield crop(image, label, number, x, y, True)

i = 0
for imageFile, labelFile, filename in files:
    print(imageFile, labelFile, filename)
    image = cv2.imread(imageFile, cv2.IMREAD_UNCHANGED)
    label = cv2.imread(labelFile, cv2.IMREAD_UNCHANGED)
    label = np.array(label > 0, dtype=np.uint8)
    number = int(filename[10:-4])

    for extractedImage, extractedLabel, filename in extractImagesLabels(image, label, number):
        typ = TYPES_DISTRIBUTION[i]
        print(f"{i: 4} / {len(TYPES_DISTRIBUTION)} - {TYPES[typ] if typ in TYPES else '': >5} - {filename}")
        i += 1
        if typ == 2:
            continue
        typ = TYPES[typ]

        cv2.imwrite(DATASET_PATH.format(typ, IMAGES) + "cyberorto_" + filename, extractedImage)
        cv2.imwrite(DATASET_PATH.format(typ, LABELS) + "cyberorto_" + filename, extractedLabel)
