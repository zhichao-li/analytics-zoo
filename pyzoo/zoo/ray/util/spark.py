import os
import sys
import subprocess


# TypeError: __init__() got an unexpected keyword argument 'auth_token' <- pip install pyspark==2.4.0 solved.
from zoo.ray.util.process import session_execute


class SparkRunner():
    def __init__(self, spark_home, java_home):
        self.spark_home = spark_home
        self.java_home = java_home
        self._prepare_spark_env()
        import pyspark
        # TODO: we need to make sure that this path is consistent with spark_home
        print("Current pyspark location is : {}".format(pyspark.__file__))


    def _prepare_spark_env(self):
        # This is a must otherwise javagateway would throw connect error
        os.environ["SPARK_HOME"] = self.spark_home
        os.environ["JAVA_HOME"] = self.java_home
        sys.path.append("%s/python/lib/pyspark.zip" % self.spark_home)
        # TODO: find the py4j verswion via wildcard
        # spark2.4.0
        sys.path.append("%s/python/lib/py4j-0.10.7-src.zip" % self.spark_home)

    def init_spark(self,
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
                   penv_archive=None,
                   hadoop_conf=None,
                   hadoop_user_name=None):
        from pyspark.sql import SparkSession
        def _common_opt():
            return '--master {} --driver-memory {} '.format(master, driver_memory)

        def _yarn_opt():
            return "--archives {}#python_env --num-executors {} --executor-cores {} --executor-memory {} --py-files {}  ".format(penv_archive, num_executor, executor_cores, executor_memory, python_zip_file)

        def _submit_opt(master):
            if "local" in master:
                return _common_opt() + 'pyspark-shell', {"spark.driver.memory": driver_memory}
            elif "yarn" in master:
                conf = {
                    "spark.scheduler.minRegisterreResourcesRatio": "1.0",
                    "spark.task.cpus": executor_cores}
                if spark_yarn_jars:
                    conf.insert("spark.yarn.archive", spark_yarn_jars)
                    return _common_opt() + _yarn_opt() + 'pyspark-shell', conf
            raise Exception("Not supported master: {}".format(master))

        submit_args, conf = _submit_opt(master)
        os.environ['PYSPARK_SUBMIT_ARGS'] = submit_args

        spark_conf = SparkSession.builder
        for key, value in conf.items():
            spark_conf.config(key=key, value=value)
        spark = spark_conf.getOrCreate()
        sc = spark.sparkContext
        sc.setLogLevel("INFO")
        return sc
        # .config(key="spark.executor.pyspark.memory", value=spark_executor_pyspark_memory)

    def run_on_local(self,
                     python_loc,
                     python_zip_file,
                     driver_memory,
                     driver_cores):
        # os.environ['PYSPARK_PYTHON'] = session_execute("which python").out
        # TODO: remove python_loc, "which python" would generate /usr/bin/python
        os.environ['PYSPARK_PYTHON'] = python_loc
        return self.init_spark(
                          python_zip_file=python_zip_file,
                          driver_memory=driver_memory,
                          driver_cores=driver_cores,
                          master="local[{}]".format(driver_cores))

    def init_spark_on_yarn(self,
                           hadoop_conf,
                           spark_yarn_jars,
                           penv_archive,
                           extra_pmodule_zip,
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

        return self.init_spark(hadoop_conf=hadoop_conf,
                          spark_yarn_jars=spark_yarn_jars,
                          penv_archive=penv_archive,
                          python_zip_file=extra_pmodule_zip,
                          num_executor=num_executor,
                          executor_cores=executor_cores,
                          executor_memory=executor_memory,
                          driver_memory=driver_memory,
                          driver_cores=driver_cores,
                          spark_executor_pyspark_memory=spark_executor_pyspark_memory,
                          master=master,
                          hadoop_user_name=hadoop_user_name)
