# Sean Wade
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import seaborn

from tensorflow.examples.tutorials.mnist import input_data

#
# ==================================================================
#

def weight_variable(shape):
  initial = tf.truncated_normal( shape, stddev=0.1 )
  return tf.Variable( initial )

def bias_variable(shape):
  initial = tf.constant( 0.1, shape=shape )
  return tf.Variable(initial)

#def bernoulli(shape, prob):
#    means = tf.ones([shape[0], shape[1]]) * prob
#    return tf.select(tf.random_uniform([shape[0], shape[1]]) - means < 0,\
#            tf.ones([shape[0], shape[1]]), tf.zeros([shape[0], shape[1]]))

def bernoulli(shape, p):
    return tf.select(tf.random_uniform(shape) < p, tf.ones(shape), tf.zeros(shape))

#
# ==================================================================
#

# Declare computation graph

scale = tf.placeholder( tf.float32, name="scale")
prob = tf.placeholder( tf.float32, name="prob")
y_ = tf.placeholder( tf.float32, shape=[None, 10], name="y_" )
x = tf.placeholder( tf.float32, [None, 784], name="x" )

W1 = weight_variable([784, 500])
b1 = bias_variable([500])
bern = bernoulli(tf.shape(W1), prob)
h1 = tf.nn.relu( tf.matmul( x, tf.mul( W1, bern ))+ b1 )
h1 = tf.mul(h1, scale)

W2 = weight_variable([500, 500])
b2 = bias_variable([500])
bern = bernoulli(tf.shape(W2), prob)
h2 = tf.nn.relu( tf.matmul( h1, tf.mul( W2, bern ) ) + b2 )
h2 = tf.mul(h2, scale)

W3 = weight_variable([500, 1000])
b3 = bias_variable([1000])
bern = bernoulli(tf.shape(W3), prob)
h3 = tf.nn.relu( tf.matmul( h2, tf.mul( W3, bern) ) + b3 )
h3 = tf.mul(h3, scale)

W4 = weight_variable([1000, 10])
b4 = bias_variable([10])
bern = bernoulli(tf.shape(W4), prob)
y_hat = tf.nn.softmax(tf.mul( tf.matmul(h3, tf.mul( W4, bern)) + b4, scale))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_hat), reduction_indices=[1]))
xent_summary = tf.scalar_summary( 'xent', cross_entropy )

correct_prediction = tf.equal(tf.argmax(y_hat,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_summary = tf.scalar_summary( 'accuracy', accuracy )

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#
# ==================================================================
#

prob_list = np.linspace(0,1,10)
train_acc = []
test_acc = []

for p in prob_list:
    sess = tf.Session()
    sess.run( tf.initialize_all_variables() )
    summary_writer = tf.train.SummaryWriter("./tf_logs", graph=sess.graph)
    merged = tf.merge_all_summaries()

    #
    # ==================================================================
    #

    # NOTE: we're using a single, fixed batch of the first 1000 images
    mnist = input_data.read_data_sets( "MNIST_data/", one_hot=True )

    images = mnist.train.images[ 0:1000, : ]
    labels = mnist.train.labels[ 0:1000, : ]

    for i in range( 1500 ):
        _, acc = sess.run( [ train_step, accuracy ], feed_dict={ x: images, y_: labels, prob:p, scale:1} )
        print( "step %d, training accuracy %g" % (i, acc) )
    train_acc.append(acc)
    

    final_acc = sess.run( accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels, prob:1, scale:p } )
    print( "test accuracy %g" % final_acc )
    test_acc.append(final_acc)

plt.title("Dropconnect Comparison")
plt.xlabel("Classification Accuracy")
plt.ylabel("Keep Probability")
plt.plot(prob_list, train_acc)
plt.plot(prob_list, test_acc)
plt.plot(prob_list, [.81]*len(prob_list), ls='--')
plt.legend(["Training", "Test", "Baseline"], loc=4)
plt.savefig("dropconnect.png")
plt.show()
#  if i%10==0:
#      tmp = sess.run( accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels } )
#      print( "          test accuracy %g" % tmp )

#
# ==================================================================
#
