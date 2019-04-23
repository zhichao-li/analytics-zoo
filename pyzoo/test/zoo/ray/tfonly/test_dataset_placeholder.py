import tensorflow as tf

# https://www.tensorflow.org/guide/datasets
# 0. For dataset, it provide batch, map ... transformer
#    and if you want to integrate it into graph model, you should `tensor = dataset.xxiterator.get_next()`
# 1. create the numpy ndarray as constant for Dataset would comsumming tons of memory as Constant in TF would be copied around
# so you would want to use the placeholder solution as described in: https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays

def build_dataset(images, labels, batch_size, train = True, dtype=tf.float32):
    images_placeholder = tf.placeholder(dtype, images.shape)
    labels_placeholder = tf.placeholder(dtype, labels.shape)
    dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))

    # transform
    dataset = dataset.batch(batch_size)
    # if without repeat(), then sess.run would throw exception at the end of loop iteration.
    # dataset = dataset.repeat()
    if train:
        dataset.shuffle(buffer_size=16 * batch_size)

    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    # sess = tf.Session()
    # sess.run(iterator.initializer, feed_dict={input_placeholder: x})
    return images, labels, iterator.initializer, images_placeholder, labels_placeholder

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:10]
y_train = y_train[:10]
batch_size = 3
num_samples = x_train.shape[0] - x_train.shape[0] % batch_size
x_train = x_train[:num_samples]
y_train = y_train[:num_samples]
images_tensor, labels_tensor, iterator_initializer, images_placeholder, labels_placeholder = build_dataset(x_train, y_train, batch_size, False)

# images or labels is a tensor, we can use it to build model
# i.e
# input1 = tf.keras.layers.Flatten(input_shape=(28, 28))(images)
# tf.keras.layers.Dense(512, activation=tf.nn.relu)(input1)

model = labels_tensor + 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # NB !! : do not put the sess.run(iterator_initializer into the itator for loop, otherwise it always print from the begining.!!!
    sess.run(iterator_initializer, feed_dict={images_placeholder: x_train, labels_placeholder: y_train})
    for i in range(3):
        image, l = sess.run([images_tensor, model])
        print("Iterator: {} label: {}".format(i, l))

