import os
import sys
import subprocess
import pyspark

print(pyspark.__file__)
from pyspark.sql import SparkSession

def init_spark(spark_home,
                       java_home,
                       python_zip_file,
                       driver_memory,
                       driver_cores,
                       master=None,
                       # settings for cluster mode
                       executor_cores=None,
                       executor_memory=None,
                       spark_executor_pyspark_memory=None,
                       #  settings for yarn only
                       num_executor=None,
                       spark_yarn_jars=None,
                       python_env_archive=None,
                       hadoop_conf=None,
                       hadoop_user_name=None):
    def _common_opt():
        # return '--master {} --py-files {} --driver-memory {} '.format(master, python_zip_file,
        #                                                               driver_memory)
        return '--master {} --driver-memory {} '.format(master, driver_memory)

    def _yarn_opt():
        return "--archives {}#python_env --num-executors {} --executor-cores {} --executor-memory {} --py-files {}  ".format(
            python_env_archive, num_executor, executor_cores, executor_memory, python_zip_file)

    def _submit_opt(master):
        if "local" in master:
            return _common_opt() + 'pyspark-shell', {"spark.driver.memory":driver_memory}
        elif "yarn" in master:
            conf = {
                    "spark.scheduler.minRegisterreResourcesRatio":"1.0",
                    "spark.task.cpus":executor_cores}

            if spark_yarn_jars:
                conf.insert("spark.yarn.archive", spark_yarn_jars)
            return _common_opt() + _yarn_opt() + 'pyspark-shell', conf
        else:
            raise Exception("Not supported master: {}".format(master))

    # SPARK_HOME = "/home/zhichao/god/zhichao/spark-2.2.0-bin-hadoop2.7/"
    os.environ["JAVA_HOME"] = java_home
    submit_opt, conf = _submit_opt(master)
    os.environ['PYSPARK_SUBMIT_ARGS'] = submit_opt


    # sys.path.append("%s/python/lib/py4j-0.10.4src.zip" % SPARK_HOME) #spark2.2.0
    # sc._conf.get("spark.driver.memory"
    # TypeError: __init__() got an unexpected keyword argument 'auth_token' <- pip install pyspark==2.4.0 solved.
    # spark_conf = SparkConf().setMaster("local[4]").set("spark.driver.memory", "2g")

    spark_conf = SparkSession.builder
    for key, value in conf.items():
        spark_conf.config(key=key, value=value)
    spark = spark_conf.getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("INFO")
    return sc, os.environ['PYSPARK_PYTHON']



    # .config(key="spark.executor.pyspark.memory", value=spark_executor_pyspark_memory)

def init_spark_on_local(spark_home,
                     java_home,
                     python_loc,
                     python_zip_file,
                     driver_memory="10g",
                     driver_cores=10):
    os.environ['PYSPARK_PYTHON'] = python_loc
    return init_spark(spark_home=spark_home,
                       java_home=java_home,
                       python_zip_file=python_zip_file,
                       driver_memory=driver_memory,
                       driver_cores=driver_cores,
                       master="local[{}]".format(driver_cores))


def init_spark_on_yarn(spark_home,
                       java_home,
                       hadoop_conf,
                       spark_yarn_jars,
                       python_env_archive,
                       python_zip_file,
                       num_executor,
                       executor_cores=28,
                       executor_memory="100g",
                       driver_memory="10g",
                       driver_cores=10,
                       spark_executor_pyspark_memory="60g",
                       master="yarn",
                       hadoop_user_name="root"):
    os.environ["HADOOP_CONF_DIR"] = hadoop_conf
    os.environ['HADOOP_USER_NAME'] = hadoop_user_name
    os.environ['PYSPARK_PYTHON'] = "python_env/bin/python"

    return init_spark(spark_home=spark_home,
                   java_home=java_home,
                   hadoop_conf=hadoop_conf,
                   spark_yarn_jars=spark_yarn_jars,
                   python_env_archive=python_env_archive,
                   python_zip_file=python_zip_file,
                   num_executor=num_executor,
                   executor_cores=executor_cores,
                   executor_memory=executor_memory,
                   driver_memory=driver_memory,
                   driver_cores=driver_cores,
                   spark_executor_pyspark_memory=spark_executor_pyspark_memory,
                   master=master,
                   hadoop_user_name=hadoop_user_name)