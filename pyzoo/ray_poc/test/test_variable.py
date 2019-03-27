
import tensorflow as tf

x = tf.Variable(4)
# we can feed data into Variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(x.eval()) # print 4
    print(sess.run(x, feed_dict={x: 15})) # print 15


