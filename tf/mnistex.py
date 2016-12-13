from __future__ import absolute_import
from __future__ import division
from Loader import *
from TFDataSet import *
import argparse
from softmax import train_soft

#Format the training images to [0, 1) size
train_images, train_labels = getSample(60000)
train_images = train_images/255.0

#Format the testing images to [0, 1) size
test_images, test_labels = getTestingSample(10000)
test_images = test_images/255.0

data_set = TFDataSet(train_images, train_labels, test_images, test_labels)

train_soft(data_set)