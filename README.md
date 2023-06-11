# Crop detection

This repository contains a keras AI model trained to detect crops and vegetation in images of vegetable-gardens ("orto" in italiano) taken perpendicular to the terrain. The model is accessible via an installable python module: `crop_detection`.

<img width="500px" src="./example_result.png"/>

## Requirements & module installation

- Install OpenCV with Python 3 support
  - `import cv2` must work without throwing exceptions
  - on Raspberry either compile from source, or `sudo apt-get install libatlas3-base` and then let `setup.py` install `opencv-python` 
- Install TensorFlow >=2.4.0
  - on Raspberry use [this repo](https://github.com/bitsy-ai/tensorflow-arm-bin)
  - Don't worry if TensorFlow and OpenCV require different versions of numpy, just make sure the latest one is installed and ignore pip warnings
- You *may* need this on raspberry: `sudo apt install libatlas-base-dev`
- The `setup.py` file can be used to **install** the module: just run `python3 -m pip install .` in the root directory
  - It will take care of installing the needed dependencies (OpenCV and TensorFlow), but on Raspberry it won't work as explained above
  - Note: `pip` may give some warnings that can be solved by appending `--use-feature=in-tree-build` to the command, but they can be ignored

## Repository file tree

- `crop_detection/` is a Python 3 module that can be imported or installed
- `converters/` contains scripts useful to build a dataset in the `dataset/` folder
- `datasets_raw/` contains the raw datasets you will download
- `datasets_raw/cyberorto/` contains images taken by MindsHub's cyberorto, along with scripts that generate labels based on pixel color heuristics (see the `generators/cyberorto_*.py` scripts)
- `dataset/` will contain, once generated, test and train data with both images and labels

## Model name

The model name is like this: `model_INPUTHEIGHTxINPUTWIDTH_DATASETVERSION_EPOCH.hd5`
- `INPUTHEIGHTxINPUTWIDTH` represents the size of the input image, e.g. `352x480`
- `DATASETVERSION` identifies what dataset used to train was composed of:
	- `1`: *ijrr_sugarbeets*, *synthetic_sugarbeat_random_weeds*
	- `2`: *ijrr_sugarbeets*, *synthetic_sugarbeat_random_weeds*, *cwfid*
	- `3`: *ijrr_sugarbeets*, *synthetic_sugarbeat_random_weeds*, *cwfid*, *cyberorto*
- `EPOCH` is the number of epochs the model was trained for, e.g. `10`

## Using the converters

Choose some datasets you want to use from the list below, then download them and unpack them in the `datasets_raw/` subfolder (which you will need to create) in the repo's root. Then you can run `python3 converters/converter_DATASET_NAME.py` to create (part of) a training dataset with both `train/` and `test/` images in the `dataset/` subfolder.

## Sources

### Datasets

- `synthetic_sugarbeat_random_weeds`
  - [Synthetic dataset - sugarbeets (380MB)](https://www.diag.uniroma1.it/~labrococo/fsd/syntheticdatasets.html)
  - [~~paper~~](https://www.diag.uniroma1.it//~pretto/papers/dpgp_IROS2017.pdf) (link not working anymore)
  - download [synthetic_sugarbeet_random_weeds.tar.gz](https://drive.google.com/uc?id=1ZcoubWc5kEV-NO3SKNLXQ5MTsPaYmof7&export=download), move it in the `datasets_raw/` subfolder and unpack it

- `ijrr_sugarbeets_2016_annotations`
  - [Bonirob dataset - sugarbeets (5TB)](https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/)
  - [website](https://www.ipb.uni-bonn.de/data/sugarbeets2016/)
  - using only `part01` is more than enough (2GB)
  - download [ijrr_sugarbeets_2016_annotations.part01.rar](https://www.ipb.uni-bonn.de/datasets_IJRR2017/annotations/cropweed/ijrr_sugarbeets_2016_annotations.part01.rar), move it in the `datasets_raw/` subfolder and unpack it with `unrar x ijrr_sugarbeets_2016_annotations.part01.rar` (it will give an error at the end about other parts of the RAR archive missing, ignore it)

- `cwfid`
  - [CWFID - Crop Weed Field Image Dataset - carrots (80MB)](https://github.com/cwfid/dataset)
  - [paper](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ECCV-2014/workshops/w23/paper26.pdf)
  - run `git clone https://github.com/cwfid/dataset` in the `datasets_raw/` subfolder

- [~~Aberystwyth Leaf Evaluation Dataset (62GB)~~](https://zenodo.org/record/168158#.WDcbSB8zpZU) not useful but still noteworthy

### Tutorials & techniques

This is the actually used tutorial and model: [Segmentation tutorial 4 (custom keras)](https://keras.io/examples/vision/oxford_pets_image_segmentation/)

#### Others (not used):

[~~SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation~~](https://arxiv.org/pdf/1511.00561.pdf)

[~~SegNet tutorial 1 (custom caffe)~~](http://mi.eng.cam.ac.uk/~agk34/demo_segnet/tutorial.html)
- [Repo with models](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/Models)
- [Caffe - Deep learning framework](http://caffe.berkeleyvision.org) - `sudo apt install caffe-cpu`

[~~SegNet tutorial 2 (keras)~~](https://github.com/0bserver07/Keras-SegNet-Basic)

[~~Segmentation tutorial 3 (keras-segmentation)~~](https://www.kaggle.com/bulentsiyah/deep-learning-based-semantic-segmentation-keras)

[~~Fast and Accurate Crop and Weed Identification with Summarized Train Sets for Precision Agriculture~~](https://www.diag.uniroma1.it/~pretto/papers/pnp_ias2016.pdf)
