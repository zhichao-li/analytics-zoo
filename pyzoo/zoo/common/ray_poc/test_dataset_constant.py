
import tensorflow as tf

# https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays

def build_dataset(x, y, batch_size, train=True):
    num_samples = x.shape[0] - x.shape[0] % batch_size
    # TODO: is there memory copy here?
    dataset = tf.data.Dataset.from_tensor_slices((x[:num_samples], y[:num_samples]))
    # what if the input is not evently divided by the batch_size ? take a look at the comments of batch:
    # by default it would generate a result with what if have, i.e [3, CHW] instead of [batch, CHW]
    dataset = dataset.batch(batch_size)
    # if without repeat(), then sess.run would throw exception at the end of loop iteration.
    # dataset = dataset.repeat()
    if train:
        dataset.shuffle(buffer_size=16 * batch_size)
    # Take a look at the comments. it would atomatically keep on getting the next batch with sess.run, not just one.
    images, labels = dataset.make_one_shot_iterator().get_next()
    # images = tf.reshape(images, [batch_size, 28, 28, 1])
    # labels = tf.reshape(labels, [batch_size, 1])
    return images, labels


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

images, labels = build_dataset(x_train[:10], y_train[:10], 3, False)

# images or labels is a tensor, we can use it to build model
# i.e
# input1 = tf.keras.layers.Flatten(input_shape=(28, 28))(images)
# tf.keras.layers.Dense(512, activation=tf.nn.relu)(input1)

model = labels + 1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3): # change to 40 would throw exception
        image, l = sess.run([images, model])
        print("Iterator: {} label: {}".format(i, l))


