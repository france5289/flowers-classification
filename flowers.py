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

''' define parameters '''
BATCH_SIZE = 50
LEARNING_RATE = 0.02
STEPS = 200
SEED = 66478

beginTime = time.time()


def truncated_variable(shape, mean, stddev):
  return tf.Variable( tf.truncated_normal(shape, mean=mean, stddev=stddev, seed=SEED) )

def constant_variable(shape, value):
  return tf.Variable( tf.constant(value=value, shape=shape) )

def random_variable(shape):
  return tf.Variable( tf.random_uniform(shape=shape, seed=SEED) )


''' Define the TensorFlow graph '''
images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3]) # NHWC
labels = tf.placeholder(tf.int64, shape=[None])


# Convolution layer 1
conv1_Channel = 8
conv1_W = truncated_variable([5, 5, 3, conv1_Channel], 0.0, 5e-2) # HWIO
conv1_b = truncated_variable([conv1_Channel], 0.0, 5e-2)

conv1 = tf.nn.conv2d( images, conv1_W, strides=[1, 1, 1, 1], padding='VALID' )
bias1 = tf.nn.bias_add(conv1, conv1_b)

pool1 = tf.nn.avg_pool(bias1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolution layer 2
conv2_Channel = 16
conv2_W = truncated_variable([5, 5, conv1_Channel, conv2_Channel], 0.0, 5e-2)
conv2_b = truncated_variable([conv2_Channel], 0.0, 5e-2)

conv2 = tf.nn.conv2d( pool1, conv2_W, strides=[1, 1, 1, 1], padding='VALID' )
bias2 = tf.nn.bias_add(conv2, conv2_b)

pool2 = tf.nn.avg_pool(bias2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Convolution layer 3
conv3_Channel = 32
conv3_W = truncated_variable([3, 3, conv2_Channel, conv3_Channel], 0.0, 5e-2)
conv3_b = truncated_variable([conv3_Channel], 0.0, 5e-2)

conv3 = tf.nn.conv2d( pool2, conv3_W, strides=[1, 1, 1, 1], padding='VALID' )
bias3 = tf.nn.bias_add(conv3, conv3_b)

pool3 = tf.nn.avg_pool(bias3, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')


# FLATTEN
'''
f_in_size = pool1.get_shape().as_list()[1]
units = f_in_size * f_in_size * conv2_Channel
flatten = tf.reshape(pool1, [-1, units])
'''
flatten = tf.layers.Flatten()(pool3) # change here: flatten input layer 
units = flatten.get_shape().as_list()[1] # [0] = batch
print(units)

# NN layer
nn1_W = truncated_variable([units, 256], 0.0, 5e-2)
nn1_b = truncated_variable([256], 0.0, 5e-2)

nn1_m = tf.matmul(flatten, nn1_W)
nn1 = tf.nn.bias_add(nn1_m, nn1_b)
relu1 = tf.nn.relu(nn1)

# NN layer
nn2_W = truncated_variable([256, 5], 0.0, 5e-2)
nn2_b = truncated_variable([5], 0.0, 5e-2)

nn2_m = tf.matmul(relu1, nn2_W)
logits = tf.nn.bias_add(nn2_m, nn2_b)


''' Learning model, training weighting '''
# Loss function sigmoid - cross entropy
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# Define the training operation
train_step = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
#train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

''' For inference and calculating the accuracy of testing data '''
# Get prediction
pred = tf.argmax(logits, 1)

# Compare prediction with true label
correct_prediction = tf.equal(pred, labels)

# Calculate the accuracy of predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


''' Run the Tensorflow graph '''
with tf.Session() as sess:
  # Initialize variables
  sess.run(tf.global_variables_initializer())

  #Prepare train data
  t1 = time.time()
  train_image, train_label, test_image, test_label = load_data.load_data(num=5000, test_size=0.1)
  t2 = time.time()
  print('Load data time use: {:5.2f}s'.format(t2 - t1))

  print('Train data images shape:', train_image.shape)
  print('Test data images shape:', test_image.shape)
  train_num = train_label.shape[0]
  test_num = test_label.shape[0]

  # Training
  for i in range(STEPS):
    # Get a batch data
    start = (i*BATCH_SIZE) % train_num
    if (start+BATCH_SIZE) > train_num:
      end = train_num
    else:
      end = start+BATCH_SIZE

    train_batch = train_image[ start : end ]
    label_batch = train_label[ start : end ]
    sess.run(train_step, feed_dict={images: train_batch, labels: label_batch})

    train_accuracy = sess.run(accuracy, feed_dict={ images: train_batch, labels: label_batch })
    print('Step {:6d}: training accuracy {:1.2f}'.format(i, train_accuracy))
    del train_batch, label_batch
    gc.collect()

  t3 = time.time()
  print('Training use: {:5.2f}s'.format(t3 - t2))


  # Testing
  test = 0.000
  for i in range(0, test_num, BATCH_SIZE):
    start = i
    if start+BATCH_SIZE > test_num:
      end = test_num
    else:
      end = start+BATCH_SIZE
    test_images = test_image[start : end]
    test_labels = test_label[start : end]
    test_accuracy = sess.run(accuracy, feed_dict={ images: test_images, labels: test_labels})
    test += (test_accuracy*(end-start))
    del test_images, test_labels
    gc.collect()

  print('Test accuracy {:1.3f}'.format( (test/test_num) ))
  t4 = time.time()
  print('Test use: {:5.2f}s'.format(t4 - t3))
  
  # Inference image
  number = random.randint(0, test_num) #input('Enter test image number: ')
  images_batch = test_image[int(number):int(number)+1]
  labels_batch = test_label[int(number):int(number)+1]
  inference = sess.run(pred, feed_dict={ images: images_batch })
  #load_data.showImage(test_image[int(number)])
  load_data.showCategory(inference.flatten()[0])
  print('True category: {}.' . format(load_data.CLASSES[labels_batch[0]]))

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))

