import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # do not use the gpu
import keras
import numpy as np
import cv2

INPUT_IMAGES = [
	"./dataset/test/images/A_000504.png",
	"./dataset/test/images/bonirob_2016-05-23-10-57-33_4_frame129.png",
	"./dataset/test/images/cwfid_001_scaled_.png",
#	"../testing_images/x.jpg",
#	"../testing_images/y.jpg",
#	"../testing_images/z.jpg",
]
CHECKPOINT_PATH = "./checkpoint_352x480_dataset2"
EPOCHS = [10]

for EPOCH in EPOCHS:
	print(f"Epoch {EPOCH}: ", end="", flush=True)
	model = keras.models.load_model(f"{CHECKPOINT_PATH}/model_{EPOCH}.hd5")
	outImage = np.zeros((352*4, 480 * len(INPUT_IMAGES), 3), dtype=np.uint8)

	for i, path in enumerate(INPUT_IMAGES):
		print(i, end=" ", flush=True)
		image = cv2.imread(path)[:352, :, :]
		out = model.predict(np.expand_dims(image, axis=0))

		out1 = np.array(out[0,:,:,1] * 255, dtype=np.uint8)
		out2 = np.array((out1 > 128) * 255, dtype=np.uint8)

		erodeDilateKernel = np.ones((2, 2), np.uint8)
		out3 = cv2.erode(out2, erodeDilateKernel, iterations=3)
		out3 = cv2.dilate(out3, erodeDilateKernel, iterations=3)

		outImage[     :352,   480*i:480*(i+1), :] = image
		outImage[352  :352*2, 480*i:480*(i+1), :] = cv2.cvtColor(out1, cv2.COLOR_GRAY2RGB)
		outImage[352*2:352*3, 480*i:480*(i+1), :] = cv2.cvtColor(out2, cv2.COLOR_GRAY2RGB)
		outImage[352*3:352*4, 480*i:480*(i+1), :] = cv2.cvtColor(out3, cv2.COLOR_GRAY2RGB)

	print("Writing", end=" ", flush=True)
	cv2.imwrite(f"outImage{EPOCH}.png", outImage)
	print("Done!")
