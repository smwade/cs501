# Sean Wade
# Lab 9: Siamese Network

import tensorflow as tf
import numpy as np
from ImageData import ImageData

relu = tf.nn.relu
# ------------------------------------------------------------------------------ 
# Global Variables

BATCH_SIZE = 128
CONV_WEIGHT_STDDEV = 0.1

# ------------------------------------------------------------------------------ 
# Helper Functions

def linear(in_var, output_size, name="linear", stddev=0.02, bias_val=0.0):
    shape = in_var.get_shape().as_list()

    with tf.variable_scope( name ):
        W = tf.get_variable( "W", [shape[1], output_size], tf.float32,
                              tf.random_normal_initializer( stddev=stddev ) )
        b = tf.get_variable( "b", [output_size],
                             initializer=tf.constant_initializer( bias_val ))

        return tf.matmul( in_var, W ) + b

def conv(x, filters_out, ksize=3, stride=1, name='conv'):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    with tf.variable_scope(name):
        W = tf.get_variable('W', shape, initializer=initializer)
        return tf.nn.conv2d( x, W, strides=[1, stride, stride, 1], padding='SAME')

def maxPool(x, ksize=3, stride=2, name='maxPool'):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, 
                ksize=[1, ksize, ksize, 1],
                strides=[1, stride, stride, 1],
                padding='SAME')

def l2norm(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x)))


# ------------------------------------------------------------------------------ 
# Model

data1 = tf.placeholder(tf.float32, shape=[None, 250, 250, 1], name="data1")
data2 = tf.placeholder(tf.float32, shape=[None, 250, 250, 1], name="data2")
label1 = tf.placeholder(tf.float32, shape=[None], name="label1")
label2 = tf.placeholder(tf.float32, shape=[None], name="label2")

def resNet(x):
    with tf.variable_scope('start'):
        x = conv(x, filters_out=64, ksize=7, stride=1)
        x = relu(x)
        x = maxPool(x, ksize=3, stride=2)
        I = x

    with tf.variable_scope('scale1'):
        for i in xrange(2):
            x = conv(x, filters_out=64, ksize=7, stride=1, name='conv{}'.format(i+1) )
            x = relu(x)

    with tf.variable_scope('Identity1'):
        I = conv(I, filters_out=64, ksize=7, stride=1)
        x = I + x
        I = x

    with tf.variable_scope('scale2'):
        for i in xrange(2):
            x = conv(x, filters_out=128, ksize=3, stride=1, name='conv{}'.format(i+1))
            x = relu(x)

    with tf.variable_scope('Identity2'):
        I = conv(I, filters_out=128, ksize=3, stride=1)
        x = I + x

    return tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")


with tf.variable_scope("Graph") as scope:
    wx1 = resNet(data1)
    scope.reuse_variables()
    wx2 = resNet(data2)
    E = l2norm(wx1-wx2)

L_g = lambda x: 0.5 * x**2
L_i = lambda x: 0.5 * (tf.maximum(0.0, 1.0 - x))**2

Y = tf.to_float(tf.equal(label1, label2))

loss = (tf.ones([BATCH_SIZE/2], dtype=tf.float32)-Y) * L_g(E) + Y * L_i(E)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
accuracy = tf.constant(.5)

# ------------------------------------------------------------------------------ 

sess = tf.Session()
sess.run(tf.initialize_all_variables())
summary_writer = tf.train.SummaryWriter('./tf_logs', graph=sess.graph)
merged = tf.merge_all_summaries()
train_acc = []
test_acc = []
print "Loading data..."
data = ImageData('./list.txt')
for i in range(500):
    print "----------"
    batch_data, batch_label = data.getBatch(BATCH_SIZE)
    batch_label = batch_label.squeeze()
    batch_data = np.expand_dims(batch_data, axis=3)
    middle = batch_data.shape[0] // 2
    _, acc = sess.run([train_step, accuracy], 
            feed_dict={ data1: batch_data[:middle],
                data2: batch_data[middle:],
                label2: batch_label[middle:],
                label1: batch_label[:middle] })
    print "Accuracy %d: %g" % (i, acc)
    train_acc.append(acc)

