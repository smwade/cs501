# Sean Wade

import tensorflow as tf
import numpy as np

from textloader import TextLoader
from tensorflow.python.ops.rnn_cell import BasicLSTMCell, MultiRNNCell, RNNCell
from tensorflow.python.ops.rnn_cell import _linear
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import seq2seq

# ------------------------------------------
#

class MyGru(RNNCell):

    def __init__(self, num_units, activation = tanh, state_is_tuple=True):
        self.num_units = num_units
        self.activation = activation

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):

            with vs.variable_scope('rz'):
                h = state
                concat = _linear([inputs, h], 2 * self.num_units, False)
                r,z = array_ops.split(1,2, concat)
                r = sigmoid(r)
                z = sigmoid(z)

            with vs.variable_scope('h_2'):
                h_tilde = self.activation(_linear([inputs, r * h], self.num_units, False))
                new_h = z * h + (1-z) * h_tilde
        
        return new_h, new_h

#
# -------------------------------------------
#
# Global variables

batch_size = 50

sequence_length = 50

data_loader = TextLoader( ".", batch_size, sequence_length )

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings

state_dim = 128

num_layers = 2

tf.reset_default_graph()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# define placeholders for our inputs.  
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='inputs' )
targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )

inputs = tf.split( 1, sequence_length, in_onehot )
inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
targets = tf.split( 1, sequence_length, targ_ph )

W = tf.Variable(tf.random_normal([state_dim, vocab_size]), name='W')
b = tf.Variable(tf.random_normal([vocab_size]), name='b')

learning_rate = 0.002

# ------------------
# COMPUTATION GRAPH 

# cell1 = BasicLSTMCell( state_dim, state_is_tuple=False )
# cell2 = BasicLSTMCell( state_dim, state_is_tuple=False )

cell1 = MyGru( state_dim, state_is_tuple=False )
cell2 = MyGru( state_dim, state_is_tuple=False )

multicell = MultiRNNCell( [cell1, cell2], state_is_tuple=True)
initial_state = multicell.zero_state(batch_size, tf.float32)

output, final_state = seq2seq.rnn_decoder(inputs, initial_state, multicell)

logits = [tf.matmul(bit, W) + b for bit in output]

one_weights = [1. for l in range(len(logits))]

loss = seq2seq.sequence_loss(logits, targets, one_weights)

optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# ------------------
# SAMPLER GRAPH 

tf.get_variable_scope().reuse_variables()

s_in_ph = tf.placeholder( tf.int32, [ 1 ], name='inputs' )
s_in_onehot = tf.one_hot( s_in_ph, vocab_size, name="input_onehot" )

s_inputs = s_in_onehot

s_initial_state = multicell.zero_state(1, tf.float32)

s_output, s_final_state = seq2seq.rnn_decoder([s_inputs], s_initial_state, multicell)

s_logits = [tf.matmul(bit, W) + b for bit in s_output]

s_probs = tf.nn.softmax(s_logits[0])

#
# ==================================================================
# ==================================================================
# ==================================================================
#

def sample( num=200, prime='ab' ):

    # prime the pump 

    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run( s_initial_state )

    # for each character, feed it into the sampler graph and
    # update the state.
    for char in prime[:-1]:
        x = np.ravel( data_loader.vocab[char] ).astype('int32')
        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel( data_loader.vocab[char] ).astype('int32')

        # plug the most recent character in...
        feed = { s_in_ph:x }
        for i, s in enumerate( s_initial_state ):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend( list(s_final_state) )

        retval = sess.run( ops, feed_dict=feed )

        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        # sample = np.argmax( s_probsv[0] )
        sample = np.random.choice( vocab_size, p=s_probsv[0] )

        pred = data_loader.chars[sample]
        ret += pred
        char = pred

    return ret

#
# ==================================================================
# ==================================================================
# ==================================================================
#
gpu_enabled = False

if gpu_enabled is True:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

else:
    sess = tf.Session()


sess.run( tf.initialize_all_variables() )
summary_writer = tf.train.SummaryWriter( "./tf_logs", graph=sess.graph )

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches

for j in range(100):

    state = sess.run( initial_state )
    data_loader.reset_batch_pointer()

    for i in range( data_loader.num_batches ):
        
        x,y = data_loader.next_batch()

        # we have to feed in the individual states of the MultiRNN cell
        feed = { in_ph: x, targ_ph: y }
        for k, s in enumerate( initial_state ):
            feed[s] = state[k]

        ops = [optim,loss]
        ops.extend( list(final_state) )

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        retval = sess.run( ops, feed_dict=feed )

        lt = retval[1]
        state = retval[2:]

        if i%1000==0:
            print "%d %d\t%.4f" % ( j, i, lt )
            lts.append( lt )

#    print sample( num=60, prime="And " )
#    print sample( num=60, prime="ababab" )
    print sample( num=60, prime=" " )
#    print sample( num=60, prime="abcdab" )

with open('final_out.txt', 'w') as outFile:
    for _ in xrange(15):
        outFile.write(sample(num=60, prime=" ")+'\n')

summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#
#plt.plot( lts )
#plt.show()
