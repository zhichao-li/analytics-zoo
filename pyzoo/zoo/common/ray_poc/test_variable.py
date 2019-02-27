
import tensorflow as tf
from zoo.common.ray_poc.lenet import Mnist

x = tf.Variable(4)
# we can feed data into Variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(x.eval()) # print 4
    print(sess.run(x, feed_dict={x: 15})) # print 15


