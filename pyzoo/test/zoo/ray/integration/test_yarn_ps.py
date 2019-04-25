from zoo.ray.benchmark.resnet import gen_resnet_fn
from zoo.ray.allreduce.sgd import DistributedEstimator
from zoo.ray.util.spark import SparkRunner
from zoo.ray.util.rayrunner import RayRunner

# for local mode
spark_home = "/home/lizhichao/bin/spark-2.4.0-bin-hadoop2.7/"
java_home = "/home/lizhichao/bin/jdk1.8.0_101/"
python_loc = "/home/lizhichao/anaconda3/envs/py36/bin/python"
python_zip_file = "/home/lizhichao/bin/god/analytics-zoo/pyzoo.zip"

# local node-018
# from ray_poc.test.allreduce.test_resnet_yarn import test_resnet_ps
#
# spark_home = "/home/zhichao/god/spark-2.4.0-bin-hadoop2.7/"
# java_home = "/home/zhichao/god/jdk1.8.0_101/"
# python_loc = "/home/zhichao/anaconda2/envs/ray36/bin/python"
# extra_pmodule_zip = "/home/zhichao/god/analytics-zoo/pyzoo.zip"


spark_runner = SparkRunner(spark_home=spark_home, java_home=java_home)
num_workers = 2
sc  = spark_runner.run_on_local(python_loc=python_loc,
                                           python_zip_file=python_zip_file,
                                           driver_memory="10g",
                                           driver_cores=num_workers + 1) # plus 1 for master

# for yarn
# spark_home = "/home/zhichao/god/spark-2.4.0-bin-hadoop2.7/"
# java_home = "/home/zhichao/god/jdk1.8.0_101/"
# hadoop_conf = "/home/zhichao/god/yarn35_config"
# extra_pmodule_zip = "/home/zhichao/god/analytics-zoo/pyzoo/zoo.zip"
# # spark_yarn_jars = None #"hdfs://172.168.2.181:9000/zhichao/ray/spark-2.4-jar.zip",
# # python_env_archive = "/home/zhichao/god/ray36.tar.gz"

# sc = spark_runner.init_spark_on_yarn(
#                                     hadoop_conf=hadoop_conf,
#                                     # penv_archive=python_env_archive,
#                                     conda_name="ray36",
#                                     extra_pmodule_zip=extra_pmodule_zip,
#                                     num_executor=num_workers + 1,
#                                     executor_cores=28,
#                                     executor_memory="100g",
#                                     driver_memory="10g",
#                                     driver_cores=10,
#                                     spark_executor_pyspark_memory="60g")
RayRunner(sc).run().start_driver()

batch_size = 64 # NB if you want to change this you need to change classic_tf_fn as well
train_fn, dataset_fn = gen_resnet_fn(batch_size)
DistributedEstimator(model_fn=train_fn, dataset_fn=dataset_fn, batch_size=batch_size, num_worker=num_workers).train(10)

import ray
ray.stop()



