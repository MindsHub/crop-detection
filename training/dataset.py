from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img
import glob
import random

def getDataset(basePath, imageSize, batchSize):
	imagePaths = glob.glob(basePath + "/images/*")
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

	return res
