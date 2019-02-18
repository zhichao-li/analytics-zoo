import tensorflow as tf
import numpy as np
import ray
from lenet import Mnist
import time

BATCH_SIZE = 100


class Network(object):
    def __init__(self, x, y):
        self.model = Mnist(x, y)
        self.loss = self.model.loss
        self.optimizer = tf.train.AdamOptimizer()
        # `compute_gradients` return a A list of (gradient, variable) pairs. Variable is always present, but
        # gradient can be `None`. so normally we only need to compute the first one.
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_ops = [g for g, v in grads_and_vars if g is not None]
        self.train_op = self.optimizer.apply_gradients(grads_and_vars) # the input should be grad and vars
        # self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        # Additional code for setting and getting the weights
        self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)
        self.sess.run(init)

    # Define a remote function that trains the network for one step and returns the
    # new weights.
    # TODO: The debuger cannot gointo here for remote function
    def apply_and_compute_grad_loss(self, weights):
        # Set the weights in the network.
        self.variables.set_weights(weights)
        # it suppose the data is from the dataset, so we don't need to feed_dict the data here.
        # Do one step of training. We only need the actual gradients so we filter over the list.
        grads_loss = self.sess.run(self.grads_ops + [self.loss])
        return grads_loss[: -1], grads_loss[-1]
        # Do one step of training.
        #_, loss_value = self.model.sess.run([self.train_op, self.model.loss])


    def get_weights(self):
        return self.variables.get_weights()

def local_test():
    x_train, y_train, x_test, y_test = Mnist.get_mnist_data()
    network = Network(x_train[:100], y_train[:100])
    grads_losses = network.apply_and_compute_grad_loss(network.get_weights())
    grads_losses

# local_test()


def ray_poc():
    import os
    import sys
    SPARK_HOME = "/home/lizhichao/bin/spark-2.1.0-bin-hadoop2.7/"
    JAVA_HOME = "/home/lizhichao/bin/jdk1.8.0_101/"
    os.environ["JAVA_HOME"] = JAVA_HOME # this is a must otherwise javagateway would throw cannot connet error
    os.environ["SPARK_HOME"] = SPARK_HOME
    # os.environ["HADOOP_CONF_DIR"] = "/opt/work/hadoop_conf/hadoop"
    # os.environ['PYSPARK_SUBMIT_ARGS'] = '--master yarn --archives /opt/work/py27.tar.gz#python_env --executor-cores 20 --executor-memory 50g --num-executors 2 pyspark-shell'
    # os.environ['PYSPARK_PYTHON'] = "python_env/bin/python"

    os.environ['PYSPARK_PYTHON'] = "/home/lizhichao/anaconda3/envs/py27/bin/python"

    sys.path.append("%s/python/lib/pyspark.zip" % SPARK_HOME)
    sys.path.append("%s/python/lib/py4j-0.10.4-src.zip" % SPARK_HOME)
    import pyspark
    print(pyspark.__file__)
    from pyspark import *
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    # spark_conf = SparkConf().setMaster("local[4]").set("spark.driver.memory", "2g")
    spark = SparkSession.builder.master("local[4]").config(key="spark.driver.memory", value="4g").getOrCreate()
    sc = spark.sparkContext



    WORKER_NUM = 4
    NUM_ITERS = 100
    x_train, y_train, x_test, y_test = Mnist.get_mnist_data()
    x_train_rdd = sc.parallelize(x_train, WORKER_NUM)
    y_train_rdd = sc.parallelize(y_train, WORKER_NUM)
    ray_master = "10.239.10.105:38202"

    ray.init(ray_master)

    x_train_ids = Mnist.save_to_ray(ray_master, x_train_rdd)
    y_train_ids = Mnist.save_to_ray(ray_master, y_train_rdd)
    print(x_train_ids)

    # Create a local model for aggregation
    # TODO: this should be a remote Actor as well for serious logic
    local_network = Network(x_test, y_test)

    local_weights = local_network.get_weights()
    # Or we can get the initial weights from remote Actor
    # weights = ray.get(actor_list[0].get_weights.remote())


    # Create actors to store the networks.
    remote_network = ray.remote(Network)
    actor_list = [remote_network.remote(x_train_ids[i], y_train_ids[i]) for i in range(WORKER_NUM)]


    # Do some steps of training.
    for iteration in range(NUM_ITERS):
        start_time = time.time()
        # Put the weights in the object store. This is optional. We could instead pass
        # the variable weights directly into step.remote, in which case it would be
        # placed in the object store under the hood. However, in that case multiple
        # copies of the weights would be put in the object store, so this approach is
        # more efficient.
        weights_id = ray.put(local_weights)
        # Call the remote function multiple times in parallel.
        gradients_loss_ids = [actor.apply_and_compute_grad_loss.remote(weights_id) for actor in actor_list]
        # Get all of the gradients.
        grads_loss_list = ray.get(gradients_loss_ids)
        gradients_list = [gl[0] for gl in grads_loss_list]
        new_losses = [gl[1] for gl in grads_loss_list]
        # sum the mean grads for all working actors.
        mean_grads = [sum([gradients[i] for gradients in gradients_list]) / len(gradients_list) for i in range(len(gradients_list[0]))]

        # mean training loss
        mean_loss = sum(new_losses) / WORKER_NUM

        # TODO: Still it only respect the first grad for now. why??
        feed_dict = {grad_op: mean_grad for (grad_op, mean_grad) in zip(local_network.grads_ops, mean_grads)}
        local_network.sess.run(local_network.train_op, feed_dict=feed_dict)
        local_weights = local_network.get_weights()

        # what's the current loss of training data? we should average the training loss and print here.
        # what's the whole throughput ?

        # All to one: the local_model responsible for 1) mean of grads, 2) update weights with the local optimizer(only need a fake data for this)
        elapsed_time = time.time() - start_time
        throughput = BATCH_SIZE * WORKER_NUM / elapsed_time
        print("Iteration {}, throughput {} loss are {}".format(iteration, throughput, mean_loss))
        # if iteration % 2 == 0:
        #     print("Iteration {}: weights are {}".format(iteration, local_weights))

ray_poc()




