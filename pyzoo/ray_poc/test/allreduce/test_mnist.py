import tensorflow as tf

from ray_poc.allreduce.sgd import RayDataSet


def classic_tf_fn():
    images = tf.keras.layers.Input((28, 28, 1))
    target = tf.keras.layers.Input((1, ))
    input1 = tf.keras.layers.Flatten(input_shape=(28, 28, 1))(images)
    dense = tf.keras.layers.Dense(512, activation=tf.nn.relu)(input1)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dropout)
    loss = tf.keras.losses.sparse_categorical_crossentropy(target, dense2)
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    return loss, optimizer, images, dense2, target


class MnistRayDataSet(RayDataSet):
    def __init__(self, batch_size):
        self.sess = tf.Session()
        self.batch_size = batch_size
        self.images, self.labels = self.get_data_set()


    # For the data input, we can either use tf.dataset or use (feed_dict together with RDD)
    def get_data_set(self):
        def get_mnist_data():
            mnist = tf.keras.datasets.mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0
            return x_train, y_train, x_test, y_test

        def build_dataset(x, y, batch_size, train=True):
            num_samples = x.shape[0] - x.shape[0] % batch_size
            dataset = tf.data.Dataset.from_tensor_slices((x[:num_samples], y[:num_samples]))
            dataset = dataset.batch(batch_size)
            # dataset = dataset.repeat()
            if train:
                dataset.shuffle(buffer_size=16 * batch_size)
            images, labels = dataset.make_one_shot_iterator().get_next()
            images = tf.reshape(images, [batch_size, 28, 28, 1])
            labels = tf.reshape(labels, [batch_size, 1])
            return images, labels

        x_train, y_train, x_test, y_test = get_mnist_data()
        images, labels = build_dataset(x_train, y_train, batch_size=self.batch_size)
        return images, labels

    # it should return list of inputs and list of labels
    # TODO: remove list here.
    def next_batch(self):
        return [[i] for i in self.sess.run([self.images, self.labels])]


from ray_poc.allreduce.sgd import DistributedOptimizer, DummyRayDataSet
from ray_poc.test.allreduce.model import create_classic_tf_model
import ray
import tensorflow as tf

def gen_dataset_fn(batch_size):
    def dataset_fn():
        return MnistRayDataSet(batch_size)
    return dataset_fn

ray.init(local_mode=True)
batch_size = 32

dopt = DistributedOptimizer.from_classic_tf(model_fn=classic_tf_fn, dataset_fn=gen_dataset_fn(batch_size), batch_size=batch_size, num_worker=2)
