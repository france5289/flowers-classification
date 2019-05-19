''' Flowers '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gc
import tensorflow as tf
import time
import sys
import os
from skimage import io
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from skimage.transform import resize
from os.path import join
root_dir = os.getcwd()
CLASSES=[]

def load_data(num, test_size):
  label_file = open(join(root_dir, 'labels.txt'), 'r')
  for line in label_file:
    l = line.split()
    CLASSES.append(l[1])
  label_file.close()
  
  datapath = join(root_dir, 'flower.record')
  print('Load data from {}'.format(datapath))
  record_iterator = tf.python_io.tf_record_iterator(path=datapath, options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))

  image_data = np.zeros((num, 224, 224, 3))
  label_data = np.zeros((num, 1))
  index = 0
  for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)

    path = bytes.decode(example.features.feature['images/path'].bytes_list.value[0])
    height = int(example.features.feature['images/height'].int64_list.value[0])
    width = int(example.features.feature['images/width'].int64_list.value[0])
    image_string = (example.features.feature['images/pixels'].bytes_list.value[0])
    label = (example.features.feature['images/label'].int64_list.value)

    try:
      image_1d = np.fromstring(image_string, dtype=np.uint8)

      image = image_1d.reshape((height, width, 3))
      image = resize(image, (224, 224, 3), mode='reflect', anti_aliasing=True)
      
      image_data[index] = image
      label_data[index] = label

      index = index+1
      if index % 100 == 0:
        print('load {:4d} data.'.format(index))

      if index >= num:
        break
      del height, width, image_string, label, image_1d, image
      gc.collect()
    except Exception:
      print('data={}, size={},{}'.format(path, height, width))
      print('Exception happened')

  image_data = image_data.reshape((-1, 224, 224, 3))
  image_data = image_data[0:index]
  label_data = label_data[0:index].astype(np.int64)
  print(image_data.shape)
  print(label_data.shape)

  test_image = image_data[0:int(index*test_size)]
  test_label = label_data[0:int(index*test_size)].flatten()
  train_image = image_data[len(test_image):index]
  train_label = label_data[len(test_label):index].flatten()

  del image_data, label_data
  gc.collect()
  return train_image, train_label, test_image, test_label


def shuffle_data( data, labels ):
  t1 = time.time()
  shuffleX = np.arange(len(data))
  np.random.shuffle(shuffleX)
  data = data[shuffleX]
  labels = labels[shuffleX]
  t2 = time.time()
  print('shuffle use: {}s'.format(t2-t1))
  return data, labels


def showImage(image):
  io.imshow(image)
  plt.show()


def showCategory(inference):
  if inference >= len(CLASSES):
    print('Image is unrecognized.')
  else:
    print('Inference category: {}.'.format(CLASSES[inference]))
