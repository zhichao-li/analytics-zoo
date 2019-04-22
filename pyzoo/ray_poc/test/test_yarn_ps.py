import sys
import os
from ray_poc.mock_user import start_ray_driver, test_sgd
from ray_poc.test.allreduce.test_resnet_yarn import test_resnet_ps
from zoo.ray.util.spark import SparkRunner

# for yarn
spark_home = "/home/zhichao/god/spark-2.4.0-bin-hadoop2.7/"
java_home = "/home/zhichao/god/jdk1.8.0_101/"
hadoop_conf = "/home/zhichao/god/yarn35_config"
spark_yarn_jars = None #"hdfs://172.168.2.181:9000/zhichao/ray/spark-2.4-jar.zip",
python_env_archive = "/home/zhichao/god/ray36.tar.gz"
extra_pmodule_zip = "/home/zhichao/god/analytics-zoo/pyzoo.zip"
python_loc = "/home/zhichao/anaconda3/envs/ray36/bin/python"


from ray_poc.rayrunner import RayRunner
spark_runner = SparkRunner(spark_home=spark_home, java_home=java_home)

num_workers = 4

sc = spark_runner.init_spark_on_yarn(
                                    hadoop_conf=hadoop_conf,
                                    penv_archive=python_env_archive,
                                    extra_pmodule_zip=extra_pmodule_zip,
                                    num_executor=num_workers,
                                    executor_cores=28,
                                    executor_memory="100g",
                                    driver_memory="10g",
                                    driver_cores=10,
                                    spark_executor_pyspark_memory="60g")

rayRunner = RayRunner(sc)

redis_address = rayRunner.run()

start_ray_driver(redis_address=redis_address)

test_resnet_ps(num_worker=num_workers)
print(redis_address)

