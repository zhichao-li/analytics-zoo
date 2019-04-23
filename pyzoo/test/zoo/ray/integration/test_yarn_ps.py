from zoo.ray.benchmark.resnet import classic_tf_fn, dataset_fn

from zoo.ray.util.spark import SparkRunner

# for yarn
spark_home = "/home/zhichao/god/spark-2.4.0-bin-hadoop2.7/"
java_home = "/home/zhichao/god/jdk1.8.0_101/"
hadoop_conf = "/home/zhichao/god/yarn35_config"
extra_pmodule_zip = "/home/zhichao/god/analytics-zoo/pyzoo/zoo.zip"
# python_loc = "/home/zhichao/anaconda3/envs/ray36/bin/python"
# spark_yarn_jars = None #"hdfs://172.168.2.181:9000/zhichao/ray/spark-2.4-jar.zip",
# python_env_archive = "/home/zhichao/god/ray36.tar.gz"


from zoo.ray.util.rayrunner import RayRunner
spark_runner = SparkRunner(spark_home=spark_home, java_home=java_home)
num_workers = 4
sc = spark_runner.init_spark_on_yarn(
                                    hadoop_conf=hadoop_conf,
                                    # penv_archive=python_env_archive,
                                    conda_name="ray36",
                                    extra_pmodule_zip=extra_pmodule_zip,
                                    num_executor=num_workers,
                                    executor_cores=28,
                                    executor_memory="100g",
                                    driver_memory="10g",
                                    driver_cores=10,
                                    spark_executor_pyspark_memory="60g")
RayRunner(sc).run().start_driver()


from zoo.ray.allreduce.sgd import DistributedOptimizer


def test_resnet_ps(num_worker):
    batch_size = 64 # NB if you want to change this you need to change classic_tf_fn as well
    dopt = DistributedOptimizer.from_classic_tf(model_fn=classic_tf_fn, dataset_fn=dataset_fn, batch_size=batch_size, num_worker=num_worker)


test_resnet_ps(num_worker=num_workers)

import ray
ray.stop()



# rayOptimizer = RayOptimizer(optimizer, loss_op, feed_data)
# rayOptimizer.minimize()


