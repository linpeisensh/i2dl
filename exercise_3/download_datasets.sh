#!/usr/bin/env bash

REL_SCRIPT_DIR=$(dirname "$0")
INITIAL_DIR=$(pwd)
cd $REL_SCRIPT_DIR
ABS_SCRIPT_DIR=$(pwd)


# Get segmentation dataset
wget http://filecremers3.informatik.tu-muenchen.de/~dl4cv/segmentation_data.zip
unzip segmentation_data.zip
rm segmentation_data.zip

# Get segmentation dataset test
wget http://filecremers3.informatik.tu-muenchen.de/~dl4cv/segmentation_data_test.zip
unzip segmentation_data_test.zip
rm segmentation_data_test.zip

# Get mnist data
wget http://filecremers3.informatik.tu-muenchen.de/~dl4cv/mnist_train.zip
unzip mnist_train.zip
rm mnist_train.zip

# Get keypoints data
wget http://filecremers3.informatik.tu-muenchen.de/~dl4cv/training.zip
unzip training.zip
rm training.zip


cd $INITIAL_DIR
