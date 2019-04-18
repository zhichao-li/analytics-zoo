import tensorflow as tf

sess = tf.Session()

m = sess.run(tf.truncated_normal((32, 10), stddev = 0.1))

a = sess.run(tf.argmax(m, 1))

a.shape

pred = tf.placeholder(dtype=tf.float32, shape=(None, 10))

result = sess.run(tf.argmax(pred, 1), feed_dict={pred:m})
result