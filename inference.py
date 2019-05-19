''' Flowers '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import gc
import load_data
import random
import os

model_path = './model'

accuracy = 0
with tf.Session() as sess:
  saver = tf.train.import_meta_graph(model_path+'/flowers-model.ckpt.meta')
  saver.restore(sess, tf.train.latest_checkpoint(model_path))
  
  graph = tf.get_default_graph()

  images = graph.get_tensor_by_name("x:0")
  labels = graph.get_tensor_by_name("y:0")
  pred = graph.get_tensor_by_name("pred:0")
  keep_prob = graph.get_tensor_by_name("keep_prob:0")

  train_image, train_label, test_image, test_label = load_data.load_data(num=100, test_size=1)
  num_test = len(test_label)

  ''' Inference image '''
  number = random.randint(0, 100)
  images_batch = test_image[int(number):int(number)+1]
  labels_batch = test_label[int(number):int(number)+1]
  inference = sess.run(pred, feed_dict={ images: images_batch, keep_prob: 1.0 })
  load_data.showCategory(inference.flatten()[0])
  print('True category: {}.' . format(load_data.CLASSES[labels_batch[0]]))
  load_data.showImage(images_batch[0])
