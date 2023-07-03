import random
from common_converters import *

IMAGES_PATH = "datasets_raw/ews/*/*_mask.png"

TYPES_DISTRIBUTION = [TRAINING] * 152 + [VALIDATION] * 38
random.Random(4).shuffle(TYPES_DISTRIBUTION)

files = []
for maskFile in glob.glob(IMAGES_PATH):
    _, filename = os.path.split(maskFile)
    filename = filename.replace("_mask", "")
    imageFile = maskFile.replace("_mask", "")
    if os.path.exists(imageFile):
        files.append((imageFile, maskFile, filename))
    else:
        print(f"image file does not exist: {imageFile}")
        exit(1)
files = sorted(files, key=lambda x: x[2])

assert len(files) == len(TYPES_DISTRIBUTION)
for i, (typ, (imageFile, maskFile, filename)) in enumerate(zip(TYPES_DISTRIBUTION, files)):
    print(f"{i: 4} / {len(TYPES_DISTRIBUTION)} - {typ} - {filename}")

    image = cv2.imread(imageFile, cv2.IMREAD_UNCHANGED)
    image = resizeImage(image)
    cv2.imwrite(DATASET_PATH.format(typ, IMAGES) + filename, image)

    mask = cv2.imread(maskFile, cv2.IMREAD_UNCHANGED)
    mask = resizeImage(mask)
    mask = np.asarray(mask[:,:,0] == 0, dtype=np.uint8)
    cv2.imwrite(DATASET_PATH.format(typ, LABELS) + filename, mask)