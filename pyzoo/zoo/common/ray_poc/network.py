import tensorflow as tf
import numpy as np
import ray
from lenet import Mnist

ray.init(redis_address="10.239.10.105:44876")

BATCH_SIZE = 100
NUM_BATCHES = 1
NUM_ITERS = 201

class Network(object):
    def __init__(self, x, y):
        self.model = Mnist(x, y)

    # Define a remote function that trains the network for one step and returns the
    # new weights.
    def step(self, weights):
        # Set the weights in the network.
        self.model.variables.set_weights(weights)
        _, loss_value = self.sess.run([self.train_op, self.loss])

        # Do one step of training.
        _, loss_value = self.sess.run([self.train_op, self.loss])
        # Return the new weights.
        return self.model.variables.get_weights()

    def get_weights(self):
        return self.model.variables.get_weights()

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
spark = SparkSession.builder.master("local[4]").config(key="spark.driver.memory", value="2g").getOrCreate()
sc = spark.sparkContext


x_train, y_train, x_test, y_test = Mnist.get_mnist_data()
x_train_rdd = sc.parallelize(x_train, 4)
y_train_rdd = sc.parallelize(y_train, 4)
ray_master = "10.239.10.105:44876"
x_train_rdd_ids = Mnist.save_to_ray(ray_master, x_train_rdd)
y_train_rdd_ids = Mnist.save_to_ray(ray_master, y_train_rdd)
print(x_train_rdd_ids)





