import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # -1 = CPU, 0 = GPU0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import numpy as np
import cv2

INPUT_IMAGES = [
	"./dataset/validation/images/A_000504.png",
	"./dataset/validation/images/bonirob_2016-05-23-10-57-33_4_frame129.png",
	"./dataset/validation/images/cwfid_001_scaled_.png",
	"./dataset/validation/images/cyberorto_001_cropped_2680_2499_3160_2851.png",
	"./dataset/training/images/FPWW0310040_RGB1_20191206_140940_6.png",
#	"./testing_images/x.jpg",
#	"./testing_images/y.jpg",
#	"./testing_images/z.jpg",
]
CHECKPOINT_PATH = "./checkpoint_352x480_dataset3"

images = np.stack([cv2.imread(path)[:352, :, :] for path in INPUT_IMAGES])

for EPOCH in range(1, 100000):
	modelPath = f"{CHECKPOINT_PATH}/model_{EPOCH}.hd5"
	imagePath = f"{CHECKPOINT_PATH}/outImage{EPOCH}.png"
	if not os.path.exists(modelPath):
		break
	if os.path.exists(imagePath):
		continue

	print(f"Epoch {EPOCH}: ", end="", flush=True)
	model = keras.models.load_model(modelPath)
	outImage = np.zeros((352 * 4, 480 * len(INPUT_IMAGES), 3), dtype=np.uint8)

	out = model.predict(images)
	for i in range(len(INPUT_IMAGES)):
		out1 = np.array(out[i,:,:,1] * 255, dtype=np.uint8)
		out2 = np.array((out1 > 128) * 255, dtype=np.uint8)

		erodeDilateKernel = np.ones((2, 2), dtype=np.uint8)
		out3 = cv2.erode(out2, erodeDilateKernel, iterations=3)
		out3 = cv2.dilate(out3, erodeDilateKernel, iterations=3)

		outImage[     :352,   480*i:480*(i+1), :] = images[i]
		outImage[352  :352*2, 480*i:480*(i+1), :] = cv2.cvtColor(out1, cv2.COLOR_GRAY2RGB)
		outImage[352*2:352*3, 480*i:480*(i+1), :] = cv2.cvtColor(out2, cv2.COLOR_GRAY2RGB)
		outImage[352*3:352*4, 480*i:480*(i+1), :] = cv2.cvtColor(out3, cv2.COLOR_GRAY2RGB)

	print("Writing", end=" ", flush=True)
	cv2.imwrite(imagePath, outImage)
	print("Done!")
