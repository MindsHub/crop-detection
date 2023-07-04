import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # -1 = CPU, 0 = GPU0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import numpy as np
import cv2
import glob

INPUT_IMAGES = [
	"./dataset/validation/images/A_000504.png",
	"./dataset/validation/images/bonirob_2016-05-23-10-57-33_4_frame129.png",
	"./dataset/validation/images/cwfid_001_scaled_.png",
	"./dataset/validation/images/cyberorto_001_cropped_2680_2499_3160_2851.png",
	"./dataset/training/images/FPWW0310040_RGB1_20191206_140940_6.png",
] + glob.glob("./training/test_images/*")

CHECKPOINT_PATH = "./checkpoint_352x480_dataset3"
H, W = 352, 480

def readImage(path):
	image = cv2.imread(path)
	h, w, c = image.shape
	return image[h//2-H//2:h//2+H//2, w//2-W//2:w//2+W//2, :]

images = np.stack([readImage(path) for path in INPUT_IMAGES])

for EPOCH in range(1, 100000):
	modelPath = f"{CHECKPOINT_PATH}/model_{EPOCH}.hd5"
	imagePath = f"{CHECKPOINT_PATH}/outImage{EPOCH}.png"
	if not os.path.exists(modelPath):
		break
	if os.path.exists(imagePath):
		continue

	model = keras.models.load_model(modelPath)
	outImage = np.zeros((H * 4, W * len(INPUT_IMAGES), 3), dtype=np.uint8)

	out = model.predict(images)
	for i in range(len(INPUT_IMAGES)):
		out1 = np.array(out[i,:,:,1] * 255, dtype=np.uint8)
		out2 = np.array((out1 > 128) * 255, dtype=np.uint8)
		out3 = out2

		erodeDilateKernel = np.ones((3, 3), dtype=np.uint8)
		out3 = cv2.dilate(out3, erodeDilateKernel, iterations=3)
		out3 = cv2.erode(out3, erodeDilateKernel, iterations=3)

		erodeDilateKernel = np.ones((2, 2), dtype=np.uint8)
		out3 = cv2.erode(out3, erodeDilateKernel, iterations=3)
		out3 = cv2.dilate(out3, erodeDilateKernel, iterations=3)

		outImage[   :H,   W*i:W*(i+1), :] = images[i]
		outImage[H  :H*2, W*i:W*(i+1), :] = cv2.cvtColor(out1, cv2.COLOR_GRAY2RGB)
		outImage[H*2:H*3, W*i:W*(i+1), :] = cv2.cvtColor(out2, cv2.COLOR_GRAY2RGB)
		outImage[H*3:H*4, W*i:W*(i+1), :] = cv2.cvtColor(out3, cv2.COLOR_GRAY2RGB)

	cv2.imwrite(imagePath, outImage)
	print("Done writing epoch", EPOCH)
