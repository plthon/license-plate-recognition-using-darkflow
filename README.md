# License Plate Recognition using Darkflow

The aim of this project is to detect license plate given an input and recognise the plate number using deep learning.

## Data Source

The data used for training and testing in this project are collected from video footage of a car dashcam.  
The dataset is from a private server of [NeounAI](https://neuon.ai/), which is not open source.

## Data Preparation

In this section, I will give a brief description of how the data is prepared to be used for training and testing.

### Frame Extraction

Use any free frame extraction software to extract frames from your dashcam footage. Either capture a frame every second or extract a total number of frames from a single footage.  
The software that I used to extract frames from the dataset is [Free Video to JPG Converter from DVDVideoSoft](https://www.dvdvideosoft.com/products/dvd/Free-Video-to-JPG-Converter.htm).

### Data Annotation using [labelImg](https://github.com/tzutalin/labelImg)

Extracted frames with visible license plate(s) were then filtered out and annotated using [labelImg](https://github.com/tzutalin/labelImg) in PascalVOC format.  
A total of 382 frames were annotated and to be used for training the deep learning model (Darkflow). 

### Model Training using [Darkflow](https://github.com/thtrieu/darkflow)

I used [Darkflow](https://github.com/thtrieu/darkflow) to train the license-plate detection model. The model is a [YOLOv2](https://arxiv.org/pdf/1612.08242.pdf) model (You Only Look Once).  
I used a pretrained model from [Theophilebuyssens](https://medium.com/@theophilebuyssens/license-plate-recognition-using-opencv-yolo-and-keras-f5bfe03afc65).
I fine-tuned the model by training the model again with our own self-annotated dataset.  
The [notebook](notebooks/LicensePlateDetection.ipynb) can be found in the notebook [directory](notebooks).

#### Training Configuration

The training configuration parameters are as followed:
- Batch size: <strong>8</strong>
- Learning rate: <strong>0.00001</strong>
- Epoch number: <strong>100</strong>

### Character Recognition

The license-plate numbers recognition is done using the [pytesseract](https://pypi.org/project/pytesseract/) library from PyPI.

### Model Testing and Results

I fed the model with 10 test images and the bounding boxes detected are compared with the ground truths to compute the mean average precision (mAP) score.  
The mAP score that I achieve is <strong>76.8%</strong> with IoU of <strong>0.3</strong>.  
<p align="center">
  <img src="https://user-images.githubusercontent.com/43836186/120340255-6a08c580-c328-11eb-8db5-a9ec1a3533c7.png">
</p>
