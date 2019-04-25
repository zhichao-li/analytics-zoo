from zoo.ray.allreduce.sgd import DistributedEstimator
import ray
import tensorflow as tf

from zoo.ray.data.lenet import MnistRayDataSet


def calc_accuracy(sess, inputs_op, outputs_op, targets_op, input_data, output_data):
    with tf.name_scope('accuracy'):
        # label [-1, 1] not one-hot encoding. If the shape mismatch, the result would be incorrect as `tf.equal` would broadcast automatically during the comparing stage.
        correct_prediction = tf.equal(tf.argmax(targets_op[0], 1),
                                      tf.cast(tf.reshape(outputs_op[0], (-1,)), tf.int64))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        return sess.run(accuracy, feed_dict={targets_op[0]: output_data, inputs_op[0]: input_data})

def model_fn():
    """
    You should add your definition here and then return (input, output, target, loss, optimizer)
    :return:
    """
    images = tf.keras.layers.Input((28, 28, 1))
    target = tf.keras.layers.Input((1, ))
    input1 = tf.keras.layers.Flatten(input_shape=(28, 28, 1))(images)
    dense = tf.keras.layers.Dense(512, activation=tf.nn.relu)(input1)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dropout)
    loss = tf.keras.losses.sparse_categorical_crossentropy(target, dense2)
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    return images, dense2, target, loss, optimizer


def gen_dataset_fn(batch_size):
    def dataset_fn():
        return MnistRayDataSet(batch_size)
    return dataset_fn
num_worker = 2
resource={"trainer": num_worker, "ps": num_worker }
ray.init(local_mode=False, resources=resource)
batch_size = 32

DistributedEstimator(model_fn=model_fn, dataset_fn=gen_dataset_fn(batch_size), batch_size=batch_size, num_worker=num_worker).train(10)
