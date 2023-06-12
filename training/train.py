import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # -1 = CPU, 0 = GPU0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' # if it worked it may help reduce memory footprint
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # necessary to solve some out of memory issues

from unet_model import unetXceptionModel
from sequence import CropWeedsSequence
from tensorflow import keras
import glob

# config
IMAGE_SIZE = (352, 480)
assert IMAGE_SIZE[0] % 32 == 0
assert IMAGE_SIZE[1] % 32 == 0
NUM_CLASSES = 2
CHECKPOINT_PATH = "./checkpoint_352x480_dataset3"

# hyperparameters (batchSize=15, learningRate=0.001) are good for
# the first 5 epochs, then use (batchSize=30, learningRate=0.0001)
BATCH_SIZE = 15
EPOCHS = 5
LEARNING_RATE = 0.001

# calculate epochs already done and load model
epochsAlreadyDone = sorted(glob.glob(CHECKPOINT_PATH + "/model_*.hd5"))
if len(epochsAlreadyDone) == 0:
	epochsAlreadyDone = 0
	print("No epochs already done, starting from the beginning")
	model = unetXceptionModel(IMAGE_SIZE, NUM_CLASSES)
	model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
else:
	epochsAlreadyDone = int(epochsAlreadyDone[-1][len(CHECKPOINT_PATH) + 7:-4])
	model = keras.models.load_model(CHECKPOINT_PATH + f"/model_{epochsAlreadyDone}.hd5")
	print(f"{epochsAlreadyDone} epochs already done, resuming")
keras.backend.set_value(model.optimizer.learning_rate, LEARNING_RATE)

# saver class that saves the model with the correct name at the end of each epoch
class CustomSaver(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		print("Saving model for epoch", epoch)
		self.model.save(CHECKPOINT_PATH + f"/model_{epoch + 1}.hd5")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# create the Sequences for loading the dataset and start training
trainSet = CropWeedsSequence(BATCH_SIZE, IMAGE_SIZE, "./dataset/train")
testSet = CropWeedsSequence(BATCH_SIZE, IMAGE_SIZE, "./dataset/validation")
print(f"Train set: {len(trainSet)} batches, {len(trainSet) * BATCH_SIZE} images\nTest set: {len(testSet)} batches, {len(testSet) * BATCH_SIZE} images")
model.fit(trainSet, validation_data=testSet, initial_epoch=epochsAlreadyDone, epochs=epochsAlreadyDone+EPOCHS, callbacks=[CustomSaver()])
