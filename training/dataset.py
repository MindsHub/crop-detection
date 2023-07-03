if __name__ == "__main__":
	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img
from tensorflow.keras import layers
from keras_cv import layers as cv_layers
import glob
import random

TRAINING_PATH = "./dataset/training"
VALIDATION_PATH = "./dataset/validation"

def dataAugmentation():
	seed = random.getrandbits(32)

	def labelLayers():
		# these ALL require the seed to be set!
		return [
			layers.RandomFlip(seed=seed),
			layers.RandomRotation((-1, 1), seed=seed),
			layers.RandomZoom((-0.2, 0.2), seed=seed),
			cv_layers.RandomShear(0.2, 0.2, seed=seed),
		]

	def imageLayers():
		return labelLayers() + [
			layers.RandomContrast(0.2),
			layers.RandomBrightness((-0.2, 0.2)),
			cv_layers.RandomSaturation((0.3, 0.6)),
			cv_layers.RandomColorDegeneration(0.2),
		]

	imageMapper = keras.Sequential(imageLayers())
	labelMapper = keras.Sequential(labelLayers())

	def augment(image, label):
		return (
			imageMapper(image, training=True),
			labelMapper(label, training=True)
		)
	return augment

def getDataset(training, batchSize):
	basePath = TRAINING_PATH if training else VALIDATION_PATH

	imagePaths = glob.glob(basePath + "/images/*")
	random.shuffle(imagePaths)
	res = tf.data.Dataset.from_tensor_slices(imagePaths)

	def imageWithLabel(imagePath):
		imagePath = imagePath.numpy().decode()
		image = np.asarray(load_img(imagePath), dtype=np.float32)

		labelPath = imagePath.replace("/images/", "/labels/")
		label = np.asarray(load_img(labelPath, color_mode="grayscale"), dtype=np.uint8)
		label = np.expand_dims(label, axis=2)

		return image, label

	res = res.map(lambda imagePath: tf.py_function(imageWithLabel, [imagePath], (np.float32, np.uint8)))
	res = res.batch(batchSize).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

	def fixup_shape(image, label):
		# https://github.com/tensorflow/tensorflow/issues/32912#issuecomment-550363802
		# without this you would get Incompatible shapes: [4] vs. [3]
		image.set_shape([None, None, None, 3])
		label.set_shape([None, None, None, 1])
		return image, label
	res = res.map(fixup_shape)

	if training:
		res = res.map(dataAugmentation())
	return res

if __name__ == "__main__":
	import cv2

	batch_size = 16
	d = getDataset(True, batch_size)
	for images, labels in d:
		break

	h, w, c = images.shape[1:]
	res = np.zeros((h * 4, w * (batch_size // 2), c), dtype=np.float32)

	for b in range(batch_size // 2):
		y = w * b
		for i, x in [(b, 0), (batch_size // 2 + b, h * 2)]:
			image, label = images[i], labels[i]
			res[x:x+h, y:y+w] = image / 255
			x += h
			res[x:x+h, y:y+w] = label

	cv2.namedWindow("Data", cv2.WINDOW_NORMAL)
	cv2.imshow("Data", res)
	while cv2.waitKey(1) & 0xFF != ord('q'): pass
	cv2.destroyAllWindows()
