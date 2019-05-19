''' Flowers '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from skimage import io
from random import shuffle

import os, sys
import numpy as np
import tensorflow as tf
import logging
import gc
import imghdr
import time

from os.path import isfile, isdir, join


flags = tf.app.flags
flags.DEFINE_bool('shuffle_imgs', True,'whether or not to shuffle images')
FLAGS = flags.FLAGS

root_dir = './'

def load_flowers_dataset(shuffle_img):
  root_lsdir = os.listdir(root_dir)
  data_dir = []
  dataset = []
  label = []
  label_file = open('./labels.txt', 'w')

  for i in root_lsdir:
    fullpath = join('./', i)
    if isdir(fullpath) and i != '__pycache__' and i != 'model' and i != 'Tensorboard' and i != '.git':
      data_dir.append( i )
      temp = { 'id': len(label),'name': i }
      label_file.write(str(len(label)) + ' ' + i + '\n')
      label.append(temp)
  label_file.close()
  print(data_dir)
  print(label)
    
  file_dir = os.listdir('./')
  i = 0
  data_num = 0
  for f in data_dir:
    fullpath = join(root_dir, f)
    image_list = os.listdir(fullpath)
    data_num += len(image_list)
    for img in image_list:
      img_path = join(fullpath, img)

      try:
        imgType = imghdr.what(img_path)
        if imgType != None:
          img = io.imread(img_path)
          img_size = img.shape

          img_info = {}
          img_info['path'] = img_path.encode('utf-8')
          img_info['height'] = img_size[0]
          img_info['width'] = img_size[1]
          img_info['pixel_data'] = img.tostring()
          img_info['labels'] = int(label[i]['id'])

          dataset.append(img_info)
          del img, img_size, img_info
          gc.collect()
      except IOError:
        print('IOError: {}'.format(img_path))
    i = i+1
    print('Folder%d load finish.'%i)

  print('data num ={}'.format(len(dataset)))
  if shuffle_img:
    np.random.shuffle(dataset)
  return dataset


def dict_to_flowers_example(img_data):
  example = tf.train.Example(features=tf.train.Features(feature={
    'images/path': bytes_feature(img_data['path']),
    'images/height': int64_feature(img_data['height']),
    'images/width': int64_feature(img_data['width']),
    'images/label': int64_feature(img_data['labels']),
    'images/pixels': bytes_feature(img_data['pixel_data']),
    #'images/format': bytes_feature('jpeg'.encode('utf-8')),
  }))
  return example


def main(_):
  t1 = time.time()
  # load total image data
  flowers_data = load_flowers_dataset(FLAGS.shuffle_imgs)
  total_imgs = len(flowers_data)
  # write data to tf record
  compression = tf.python_io.TFRecordCompressionType.GZIP
  
  # record file name
  file_name_out = 'flower.record'
  with tf.python_io.TFRecordWriter(file_name_out, options=tf.python_io.TFRecordOptions(compression)) as tfrecord_writer:
    for index, img_data in enumerate(flowers_data):
      if (index+1) % 100 == 0 or (index+1) == total_imgs:
        print("Converting images: %d / %d" % (index+1, total_imgs))

      example = dict_to_flowers_example(img_data)
      tfrecord_writer.write(example.SerializeToString())
  t2 = time.time()
  print('Write record use: {:5.2f}s' . format(t2-t1))



def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


if __name__ == "__main__":
    tf.app.run()

