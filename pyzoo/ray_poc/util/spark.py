import os
import sys

import pyspark

print(pyspark.__file__)
from pyspark.sql import SparkSession


def init_spark_local(spark_home,
                     java_home,
                     python_env_archive,
                     python_zip_file,
                     driver_memory="10g",
                     driver_cores=10):
    init_spark(spark_home=spark_home,
                       java_home=java_home,
                       python_env_archive=python_env_archive,
                       python_zip_file=python_zip_file,
                       driver_memory=driver_memory,
                       driver_cores=driver_cores)
    pass


def init_spark_on_yarn(spark_home,
                       java_home,
                       hadoop_conf,
                       spark_yarn_jars,
                       python_env_archive,
                       python_zip_file,
                       num_executor=3,
                       executor_cores=28,
                       executor_memory="100g",
                       driver_memory="10g",
                       driver_cores=10,
                       spark_executor_pyspark_memory="60g",
                       master="yarn",
                       hadoop_user_name="root"):
    pass

def init_spark(spark_home,
                       java_home,
                       hadoop_conf,
                       spark_yarn_jars,
                       python_env_archive,
                       python_zip_file,
                       num_executor=3,
                       executor_cores=28,
                       executor_memory="100g",
                       driver_memory="10g",
                       driver_cores=10,
                       spark_executor_pyspark_memory="60g",
                       master="yarn",
                       hadoop_user_name="root"):
    def _common_opt():
        return '--master {} --py-files {} --driver-memory {} '.format(master, python_zip_file,
                                                                      driver_memory)

    def yarn_opt():
        return "--archives {}#python_env --num-executors {} --executor-cores {} --executor-memory {} ".format(
            python_env_archive, num_executor, executor_cores, executor_memory)

    def submit_opt(master):
        if "local" in master:
            return _common_opt() + 'pyspark-shell'
        elif "yarn" in master:
            return _common_opt() + yarn_opt() + 'pyspark-shell'
        else:
            raise Exception("Not supported master: {}".format(master))

    # SPARK_HOME = "/home/zhichao/god/zhichao/spark-2.2.0-bin-hadoop2.7/"
    os.environ["JAVA_HOME"] = java_home
    # this is a must otherwise javagateway would throw cannot connet error
    os.environ["SPARK_HOME"] = spark_home
    os.environ["HADOOP_CONF_DIR"] = hadoop_conf

    os.environ[
        'PYSPARK_SUBMIT_ARGS'] = '--master {}  --py-files {}   --num-executors {}  pyspark-shell'.format(
        master, python_env_archive, python_zip_file, num_executor, executor_cores, executor_memory)
    os.environ['PYSPARK_PYTHON'] = "python_env/bin/python"
    os.environ['HADOOP_USER_NAME'] = hadoop_user_name
    sys.path.append("%s/python/lib/pyspark.zip" % spark_home)
    # TODO: wildcard
    sys.path.append("%s/python/lib/py4j-0.10.7-src.zip" % spark_home)  # spark2.4.0
    # sys.path.append("%s/python/lib/py4j-0.10.4src.zip" % SPARK_HOME) #spark2.2.0
    # sc._conf.get("spark.driver.memory"
    # TypeError: __init__() got an unexpected keyword argument 'auth_token' <- pip install pyspark==2.4.0 solved.
    # spark_conf = SparkConf().setMaster("local[4]").set("spark.driver.memory", "2g")
    spark = SparkSession.builder.config(key="spark.driver.memory", value=driver_memory).config(
        key="spark.yarn.archive", value=spark_yarn_jars) \
        .config(key="spark.scheduler.minRegisterreResourcesRatio", value="1.0").config(
        key="spark.task.cpus", value=str(executor_cores)) \
        .getOrCreate()
    sc = spark.sparkContext
    # sc.setLogLevel("DEBUG")
    return sc, os.environ['PYSPARK_PYTHON']



    # .config(key="spark.executor.pyspark.memory", value=spark_executor_pyspark_memory)
