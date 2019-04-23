
import tensorflow as tf
import numpy as np

v = tf.Variable(np.ones([4]), dtype=tf.int32)
# we can feed data into Variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(v.eval()) # print 4
    print(sess.run(v, feed_dict={v: np.ones([4]) + 5})) # print 15
    print(v.eval()) # if without assign and feed the data directly, the value of variable would not be changed !!!!
    p=tf.placeholder(
        tf.int32,
        [4],
        name="Placeholder_a" )
    y_assigned = v.assign(p)
    print(sess.run(y_assigned, feed_dict={p: np.ones([4]) + 5})) # print 15
    print(v.eval(sess))


