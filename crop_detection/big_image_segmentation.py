from tensorflow import keras
import numpy as np
import cv2
import os

_DEFAULT_MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + "/model_352x480_2_10.hd5"
_models = {}

def _getModel(modelPath):
	global _models
	if modelPath not in _models:
		_models[modelPath] = keras.models.load_model(modelPath)
	return _models[modelPath]

def segmentImage(imageOrImagePath, modelPath=_DEFAULT_MODEL_PATH, padding=20):
	if isinstance(imageOrImagePath, (str, bytes, os.PathLike,)):
		image = cv2.imread(imageOrImagePath)
		if image is None:
			raise ValueError(f"Image not found: {imageOrImagePath}")
	else:
		image = np.asarray(imageOrImagePath)
		if (image.dtype != np.uint8):
			raise ValueError(f"\"imageOrImagePath\" image dtype is not uint8: {image.dtype}")

	model = _getModel(modelPath)
	modelInputShape = model.layers[0].input_shape[0][1:3]
	modelOutputShape = model.layers[-1].output_shape[1:3]
	modelHeight, modelWidth = modelOutputShape
	if padding * 2 >= min(modelHeight, modelWidth):
		raise ValueError(f"The padding {padding} is too big for model shape {modelOutputShape}")

	imageShape = np.shape(image)[:2]
	imageHeight, imageWidth = imageShape
	if imageHeight < modelHeight or imageWidth < modelWidth:
		raise ValueError(f"The image of shape {imageShape} is too small for model shape {modelOutputShape}")

	# yields (xFrom, yFrom, pLeft, pTop, pRight, pBottom)
	# xFrom, yFrom is the upper left corner of the image of size modelOutputShape to submit to the model
	# pLeft, pTop, pRight, pBottom represent the padding to throw away from the model output on the four sides
	def segmentationRectanglesCoordinates():
		for x in range(0, imageWidth-padding, modelWidth-padding*2):
			for y in range(0, imageHeight-padding, modelHeight-padding*2):
				if x == 0:
					xFrom = 0
					pLeft = 0
					pRight = padding
				elif x + modelWidth >= imageWidth:
					xFrom = imageWidth - modelWidth
					pLeft = padding
					pRight = 0
				else:
					xFrom = x
					pLeft = padding
					pRight = padding

				if y == 0:
					yFrom = 0
					pTop = 0
					pBottom = padding
				elif y + modelHeight >= imageHeight:
					yFrom = imageHeight - modelHeight
					pTop = padding
					pBottom = 0
				else:
					yFrom = y
					pTop = padding
					pBottom = padding

				yield (xFrom, yFrom, pLeft, pTop, pRight, pBottom)

	result = np.empty(imageShape, dtype=np.uint8)
	for xFrom, yFrom, pLeft, pTop, pRight, pBottom in segmentationRectanglesCoordinates():
		print(xFrom, yFrom, pLeft, pTop, pRight, pBottom)
		modelInput = image[yFrom:yFrom+modelHeight, xFrom:xFrom+modelWidth, :]
		if modelInputShape != modelOutputShape:
			modelInput = cv2.resize(modelInput, (modelInputShape[1], modelInputShape[0]))
		modelOutput = model.predict(np.expand_dims(modelInput, axis=0))

		result[yFrom+pTop:yFrom+modelHeight-pBottom, xFrom+pLeft:xFrom+modelWidth-pRight] \
			= modelOutput[0, pTop:modelHeight-pBottom, pLeft:modelWidth-pRight, 1] * 255

	return result

if __name__ == "__main__":
	IMAGES = ["bigimage1.png", "bigimage2.jpg", "bigimage3.png", "bigimage4.jpg", "bigimage5.png"]
	for image in IMAGES:
		segmented = segmentImage(
			imageOrImagePath="../training/testing_images/" + image,
			modelPath="../training/checkpoint_352x480_dataset2/model_7.hd5",
			padding=20
		)

		cv2.imwrite("bis_" + image.replace(".jpg", ".png"), segmented)
