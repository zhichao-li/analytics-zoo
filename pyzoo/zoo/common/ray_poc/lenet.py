
import ray
import tensorflow as tf
import numpy as np

class Mnist:

    @staticmethod
    def get_mnist_data():
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return x_train, y_train, x_test, y_test

    @staticmethod
    def save_to_ray(master_addr, x_rdd_ndarray):
        def f(splitIndex, iterator):
            import ray
            ray.shutdown()
            ray.init(master_addr)
            # big ndarray sample
            big_ndarray = np.stack([i for i in iterator], axis=0)
            yield ray.put(big_ndarray)  #(splitIndex, ray.put(big_ndarray))  # performance?

        return x_rdd_ndarray.mapPartitionsWithIndex(f).collect()


    def __init__(self, x_train, y_train):
        images, labels = self.build_dataset(x_train, y_train, 32)
        input1 = tf.keras.layers.Flatten(input_shape=(28, 28))(images)
        dense = tf.keras.layers.Dense(512, activation=tf.nn.relu)(input1)
        dropout = tf.keras.layers.Dropout(0.2)(dense)
        dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(dropout)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, dense2)
        self.loss = tf.reduce_mean(loss)

        #         model.inputs = images
        #         model.targets = labels
        #         model.compile(optimizer='adam',
        #                   loss='sparse_categorical_crossentropy',
        #                   metrics=['accuracy'])


        #         # Define the loss.
        #         self.loss = model.total_loss
        #         self.inputs = model.inputs + model.targets
        #         optimizer = tf.train.GradientDescentOptimizer(0.5)
        #         self.grads = optimizer.compute_gradients(self.loss)
        #         self.train = optimizer.apply_gradients(self.grads)
        # Define the weight initializer and session.

    def build_dataset(self, x, y, batch_size, train=True):
        num_samples = x.shape[0] - x.shape[0] % batch_size
        dataset = tf.data.Dataset.from_tensor_slices((x[:num_samples], y[:num_samples]))
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        if train:
            dataset.shuffle(buffer_size=16 * batch_size)
        images, labels = dataset.make_one_shot_iterator().get_next()
        images = tf.reshape(images, [batch_size, 28, 28, 1])
        labels = tf.reshape(labels, [batch_size, 1])
        return images, labels

    def train(self):
        for i in range(20):
            _, loss_value = self.sess.run([self.train_op, self.loss])
            print("Iter: {}, Loss: {}".format(i, loss_value))


# mnist.sess.run(mnist.train, feed_dict={i: d for i, d in zip(mnist.inputs, [x_train, y_train])})
x_train, y_train, x_test, y_test = Mnist.get_mnist_data()
mnist = Mnist(x_train[:100], y_train[:100])
# # mnist.train()
# weights = mnist.variables.get_weights()
# ray_master = "10.239.10.105:44876"
# import ray
# ray.shutdown()
# ray.init(ray_master)
# wid = ray.put(weights)
# wfr = ray.get(wid)
# weights

# optimizer = tf.train.AdamOptimizer()
# grads_ops = optimizer.compute_gradients(mnist.loss)
# train_op = optimizer.apply_gradients(grads_ops)
# # self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
# init = tf.global_variables_initializer()
# sess = tf.Session()
# # Additional code for setting and getting the weights
# variables = ray.experimental.TensorFlowVariables(mnist.loss, sess)
# sess.run(init)
# result = sess.run([grads_op[0] for grads_op in grads_ops] + [mnist.loss])
# result





