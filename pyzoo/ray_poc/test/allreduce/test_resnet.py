import tensorflow as tf

from ray_poc.allreduce.sgd import RayDataSet
from ray.experimental.sgd.tfbench import model_config

class MockDataset():
    name = "synthetic"

batch_size = 64
image_shape = [batch_size, 224, 224, 3]
labels_shape = [batch_size]

# copy from ray.experimental.sgd.tfbench.test_model.py
def classic_tf_fn():
    # Synthetic image should be within [0, 255].
    inputs = tf.placeholder(
        shape=image_shape,
        dtype=tf.float32,
        name='synthetic_images')

    labels = tf.placeholder(
        shape=labels_shape,
        dtype=tf.int32,
        name='synthetic_labels')

    model = model_config.get_model_config("resnet50", MockDataset())
    logits, aux = model.build_network(
        inputs, data_format="NHWC") # TODO: check here
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

    # Implement model interface
    loss = tf.reduce_mean(loss, name='xentropy-loss')
    optimizer = tf.train.GradientDescentOptimizer(1e-6)

    return loss, optimizer, inputs, logits, labels


class ResNetDummyDataSet(RayDataSet):
    def __init__(self):
        self.sess = tf.Session()
        inputs = tf.truncated_normal(
            image_shape,
            dtype=tf.float32,
            mean=127,
            stddev=60,
            name='synthetic_images')

        labels = tf.random_uniform(
            labels_shape,
            minval=0,
            maxval=999,
            dtype=tf.int32,
            name='synthetic_labels')
        self.dummy = [[i] for i in self.sess.run([inputs, labels])]


    def next_batch(self):
        return self.dummy



from ray_poc.allreduce.sgd import DistributedOptimizer, DummyRayDataSet
from ray_poc.test.allreduce.model import create_classic_tf_model
import ray
import tensorflow as tf

def dataset_fn():
    return ResNetDummyDataSet()

ray.init(local_mode=False, log_to_driver=True)


dopt = DistributedOptimizer.from_classic_tf(model_fn=classic_tf_fn, dataset_fn=dataset_fn, batch_size=batch_size, num_worker=2)
