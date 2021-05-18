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