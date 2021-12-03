# tensorflow-resnet50-example
## Introduction

This repository is dedicated to provide source code for training :
* Build and run a resnet50 on MPPA(R) from scratch

## External package
A partial validation dataset (at least 1000 images) of 
ImageNet (ILSVRC-2012) is required for the calibration at quantization step.
You need to register on http://www.image-net.org/download-images in order 
to  get the link to download the dataset.

## Contents
You should find for the training:
* images dir: some images to test inference
* network.yaml: basic configuration file to generate model with KaNN(TM)

Python scripts as tensorflow tools:
* import_frozen_model_to_tensorboard.py : python script to create a log DIR
for tensorboard visualization
* summary_graph.py: parse nodes and count the number of nodes in pb graph

Preliminary scripts to build models:
* resnet50.py: build a resnet50(imagenet) saved model in tensorflow 2.1
* freeze_model.py: freeze a saved model in protobuf file (.pb)
* convert_tf_to_tflite.py: convert a saved mode to tflite with calibration
(required: ILSVRC2012 validation set)

Scripts to run inference with models:  
* run_saved_model_inference.py: load and get prediction from a saved_model
* run_frozen_model_inference.py: load and get prediction from a frozen graph (pb)
* run_tflite_inference.py: load and get prediction from a tflite model

## Requirements
  * CPU (x86) / K200 PCIe (ACE-acceleration)
  * MPPA(R) Accesscore release ACE_4.6.0
    * KaNN(TM), 4.6.0