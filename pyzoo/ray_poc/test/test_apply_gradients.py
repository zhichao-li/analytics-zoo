import tensorflow as tf
import numpy as np



opt = tf.train.RMSPropOptimizer(0.001)

grads = tf.placeholder(
                tf.float32,
                [34],
                name="Placeholder_grads")

weights = tf.Variable(
                initial_value=np.ones([34]),
                dtype=tf.float32,
                name="variable_weights")

vgrads = np.ones([34])

apply_op = opt.apply_gradients([(grads, weights)])

sess = tf.Session(
    config=tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1))
sess.run(tf.global_variables_initializer())

_, w = sess.run([apply_op, weights], feed_dict={grads:vgrads})

w
