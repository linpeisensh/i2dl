#!/usr/bin/env bash

REL_SCRIPT_DIR=$(dirname "$0")
INITIAL_DIR=$(pwd)
cd $REL_SCRIPT_DIR
ABS_SCRIPT_DIR=$(pwd)

cd datasets

# Get segmentation dataset
curl http://filecremers3.informatik.tu-muenchen.de/~dl4cv/segmentation_data.zip -OL segmentation_data.zip
tar -xzvf segmentation_data.zip
rm segmentation_data.zip

# Get segmentation dataset test
curl http://filecremers3.informatik.tu-muenchen.de/~dl4cv/segmentation_data_test.zip -OL segmentation_data_test.zip
tar -xzvf segmentation_data_test.zip 
rm segmentation_data_test.zip

# Get mnist data
curl http://filecremers3.informatik.tu-muenchen.de/~dl4cv/mnist_train.zip -OL mnist_train.zip 
tar -xzvf mnist_train.zip
rm mnist_train.zip

# Get keypoints data
curl http://filecremers3.informatik.tu-muenchen.de/~dl4cv/training.zip -OL training.zip
tar -xzvf training.zip
rm training.zip

cd $INITIAL_DIR
