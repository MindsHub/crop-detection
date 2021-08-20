import keras
import numpy as np
from keras.preprocessing.image import load_img
import glob
import random

class CropWeedsSequence(keras.utils.Sequence):
	def __init__(self, batchSize, imageSize, basePath):
		self.batchSize = batchSize
		self.imageSize = imageSize
		self.images = glob.glob(basePath + "/images/*")
		random.shuffle(self.images)

	def __len__(self):
		return len(self.images) // self.batchSize

	def __getitem__(self, index):
		i = index * self.batchSize
		batchImages = self.images[i:i+self.batchSize]

		x = np.zeros((self.batchSize,) + self.imageSize + (3,), dtype="float32")
		for j, path in enumerate(batchImages):
			img = np.asarray(load_img(path))
			x[j] = img[:self.imageSize[0], :self.imageSize[1], :]

		y = np.zeros((self.batchSize,) + self.imageSize + (1,), dtype="uint8")
		for j, path in enumerate(batchImages):
			path = path.replace("/images/", "/labels/")
			img = np.asarray(load_img(path, color_mode="grayscale"))
			y[j] = np.expand_dims(img[:self.imageSize[0], :self.imageSize[1]], axis=2)

		return x, y

