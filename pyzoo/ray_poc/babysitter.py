import sys
import os



spark_home = "/home/zhichao/god/spark-2.4.0-bin-hadoop2.7/"
java_home = "/home/zhichao/god/jdk1.8.0_101/"
hadoop_conf = "/home/zhichao/god/yarn55_config/etc/hadoop",
spark_yarn_jars = "hdfs://172.168.2.181:9000/zhichao/ray/spark-2.4-jar.zip",
python_env_archive = "/home/zhichao/god/ray_35.tar.gz",
python_zip_file = "/home/zhichao/god/analytics-zoo/pyzoo/ray_poc.zip"
#
# sc, python_loc = init_spark_on_yarn()

# for local mode
# spark_home = "/home/lizhichao/bin/spark-2.4.0-bin-hadoop2.7/"
# java_home = "/home/lizhichao/bin/jdk1.8.0_101/"
# python_loc = "/home/lizhichao/anaconda3/envs/py36/bin/python"
# python_zip_file = "/home/lizhichao/bin/god/analytics-zoo/pyzoo/ray_poc.zip"


def init_spark_env(spark_home):
    # this is a must otherwise javagateway would throw cannot connet error
    os.environ["SPARK_HOME"] = spark_home
    sys.path.append("%s/python/lib/pyspark.zip" % spark_home)
    # TODO: wildcard
    sys.path.append("%s/python/lib/py4j-0.10.7-src.zip" % spark_home)  # spark2.4.0

init_spark_env(spark_home) # this should happen before rayrunner

from ray_poc.rayrunner import RayRunner
from ray_poc.util.spark import init_spark_on_yarn, init_spark_on_local


# sc, python_loc = init_spark_on_local(spark_home=spark_home,
#                      java_home=java_home,
#                      python_loc=python_loc,
#                      python_zip_file=python_zip_file,
#                      driver_memory="10g",
#                      driver_cores=4)


sc, python_loc = init_spark_on_yarn(spark_home=spark_home,
                       java_home=java_home,
                       hadoop_conf=hadoop_conf,
                       spark_yarn_jars=spark_yarn_jars,
                       python_env_archive=python_env_archive,
                       python_zip_file=python_zip_file,
                       num_executor=4,
                       executor_cores=28,
                       executor_memory="100g",
                       driver_memory="10g",
                       driver_cores=10,
                       spark_executor_pyspark_memory="60g")

rayRunner = RayRunner(sc, python_loc)

redis_address = rayRunner.run()
# redis_address is 56 but it returnning 59

print(redis_address)

# TODO: setting the python log? or just print or redirect to elsewhere?
