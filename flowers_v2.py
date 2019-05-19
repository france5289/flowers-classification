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
if not os.path.isdir(model_path):
    os.mkdir(model_path)


''' define parameters '''
BATCH_SIZE = 70
LEARNING_RATE = 0.02
STEPS = 10000
SEED = 66478

beginTime = time.time()


def truncated_variable(shape, mean, stddev):
  return tf.Variable( tf.truncated_normal(shape, mean=mean, stddev=stddev, seed=SEED) )

def constant_variable(shape, value):
  return tf.Variable( tf.constant(value=value, shape=shape) )

def random_variable(shape):
  return tf.Variable( tf.random_uniform(shape=shape, seed=SEED) )


''' Define the TensorFlow graph '''
images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='x') # NHWC
labels = tf.placeholder(tf.int64, shape=[None], name='y')


# Convolution layer 1
conv1_Channel = 8
conv1_W = truncated_variable([3, 3, 3, conv1_Channel], 0.0, 5e-2) # HWIO
conv1_b = truncated_variable([conv1_Channel], 0.0, 5e-2)

conv1 = tf.nn.conv2d( images, conv1_W, strides=[1, 1, 1, 1], padding='SAME' )
bias1 = tf.nn.bias_add(conv1, conv1_b)

# Convolution layer 1-1
conv1_1_Channel = 16
conv1_1_W = truncated_variable([3, 3, conv1_Channel, conv1_1_Channel], 0.0, 5e-2) # HWIO
conv1_1_b = truncated_variable([conv1_1_Channel], 0.0, 5e-2)

conv1_1 = tf.nn.conv2d( bias1, conv1_1_W, strides=[1, 1, 1, 1], padding='SAME' )
bias1_1 = tf.nn.bias_add(conv1_1, conv1_1_b)

pool1 = tf.nn.max_pool(bias1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolution layer 2
conv2_Channel = 32
conv2_W = truncated_variable([3, 3, conv1_1_Channel, conv2_Channel], 0.0, 5e-2)
conv2_b = truncated_variable([conv2_Channel], 0.0, 5e-2)

conv2 = tf.nn.conv2d( pool1, conv2_W, strides=[1, 1, 1, 1], padding='SAME' )
bias2 = tf.nn.bias_add(conv2, conv2_b)

# Convolution layer 2-1
conv2_1_Channel = 32
conv2_1_W = truncated_variable([3,3, conv2_Channel, conv2_1_Channel], 0.0, 5e-2)
conv2_1_b = truncated_variable([conv2_1_Channel], 0.0, 5e-2)

conv2_1 = tf.nn.conv2d(bias2, conv2_1_W, strides=[1,1,1,1], padding='SAME')
bias2_1 = tf.nn.bias_add(conv2_1, conv2_1_b)


pool2 = tf.nn.max_pool(bias2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Convolution layer 3
conv3_Channel = 64
conv3_W = truncated_variable([3, 3, conv2_Channel, conv3_Channel], 0.0, 5e-2)
conv3_b = truncated_variable([conv3_Channel], 0.0, 5e-2)

conv3 = tf.nn.conv2d( pool2, conv3_W, strides=[1, 1, 1, 1], padding='SAME' )
bias3 = tf.nn.bias_add(conv3, conv3_b)

pool3 = tf.nn.max_pool(bias3, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

# Convolution layer 4
conv4_Channel = 128
conv4_W = truncated_variable([3,3, conv3_Channel, conv4_Channel], 0.0, 5e-2)
conv4_b = truncated_variable([conv4_Channel], 0.0, 5e-2)

conv4 = tf.nn.conv2d(pool3, conv4_W, strides=[1,1,1,1], padding='SAME')
bias4 = tf.nn.bias_add(conv4, conv4_b)

pool4 = tf.nn.max_pool(bias4, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

# FLATTEN
'''
f_in_size = pool1.get_shape().as_list()[1]
units = f_in_size * f_in_size * conv2_Channel
flatten = tf.reshape(pool1, [-1, units])
'''
flatten = tf.layers.Flatten()(pool4) # change here: flatten input layer 
units = flatten.get_shape().as_list()[1] # [0] = batch
print(units)

# NN layer
nn1_W = truncated_variable([units, 1024], 0.0, 5e-2)
nn1_b = truncated_variable([1024], 0.0, 5e-2)

nn1_m = tf.matmul(flatten, nn1_W)
nn1 = tf.nn.bias_add(nn1_m, nn1_b)
relu1 = tf.nn.relu(nn1)
#Dropout
keep_prob = tf.placeholder(tf.float32)
relu1_drop = tf.nn.dropout(relu1, keep_prob)
# NN layer
nn2_W = truncated_variable([1024, 512], 0.0, 5e-2)
nn2_b = truncated_variable([512], 0.0, 5e-2)

nn2_m = tf.matmul(relu1_drop, nn2_W)
nn2 = tf.nn.bias_add(nn2_m, nn2_b)
relu2 = tf.nn.relu(nn2)
# Dropout
relu2_drop = tf.nn.dropout(relu2, keep_prob)

# NN layer
nn3_W = truncated_variable([512, 5], 0.0, 5e-2) 
nn3_b = truncated_variable([5], 0.0, 5e-2)

nn3_m = tf.matmul(relu2_drop, nn3_W)
logits = tf.nn.bias_add(nn3_m, nn3_b)


''' Learning model, training weighting '''
# Loss function sigmoid - cross entropy
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
tf.summary.scalar('loss', loss)

# Define the training operation
train_step = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
#train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
#train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
#train_step = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, momentum=0.9).minimize(loss)
''' For inference and calculating the accuracy of testing data '''
# Get prediction
pred = tf.argmax(logits, 1, name='pred')

# Compare prediction with true label
correct_prediction = tf.equal(pred, labels)

# Calculate the accuracy of predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)


merged = tf.summary.merge_all()
filewriter = tf.summary.FileWriter('Tensorboard',tf.Session().graph)

''' Run the Tensorflow graph '''
with tf.Session() as sess:
  # Initialize variables
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()

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
    sess.run(train_step, feed_dict={images: train_batch, labels: label_batch, keep_prob:0.5})

    train_accuracy = sess.run(accuracy, feed_dict={ images: train_batch, labels: label_batch, keep_prob:0.5 })
    print('Step {:6d}: training accuracy {:1.2f}'.format(i, train_accuracy))
    result = sess.run(merged, feed_dict={ images: train_batch, labels: label_batch, keep_prob:1.0 })
    filewriter.add_summary(result, i)
    del train_batch, label_batch
    gc.collect()

  saver.save(sess, model_path+'/flowers-model.ckpt')

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
    test_accuracy = sess.run(accuracy, feed_dict={ images: test_images, labels: test_labels, keep_prob:1.0})
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
  inference = sess.run(pred, feed_dict={ images: images_batch, keep_prob:1.0 })
  #load_data.showImage(test_image[int(number)])
  load_data.showCategory(inference.flatten()[0])
  print('True category: {}.' . format(load_data.CLASSES[labels_batch[0]]))

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))

