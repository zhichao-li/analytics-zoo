# How to run Ray On Yarn

1) You should install Conda first and create a conda-env named "ray36"

2) Install some essential dependencies on the conda env
pip install BigDL
pip install ray[debug]
pip install conda-pack
pip install psutil

3) Download JDK8
4) DownLoad Spark-2.4
5) export PYTHONPATH=/home/lizhichao/bin/god/analytics-zoo/pyzoo/zoo.zip:$PYTHONPATH
6) start jupyter notebook and run the following code:

``` python
spark_home = "/home/zhichao/god/spark-2.4.0-bin-hadoop2.7/"
java_home = "/home/zhichao/god/jdk1.8.0_101/"
hadoop_conf = "/home/zhichao/god/yarn35_config"
extra_pmodule_zip = "/home/zhichao/god/analytics-zoo/pyzoo/zoo.zip"

from zoo.ray.util.rayrunner import RayRunner
spark_runner = SparkRunner(spark_home=spark_home, java_home=java_home)
num_workers = 4
sc = spark_runner.init_spark_on_yarn(
                                    hadoop_conf=hadoop_conf,
                                    conda_name="ray36",
                                    extra_pmodule_zip=extra_pmodule_zip,
                                    num_executor=num_workers,
                                    executor_cores=28,
                                    executor_memory="100g",
                                    driver_memory="10g",
                                    driver_cores=10,
                                    spark_executor_pyspark_memory="60g")
RayRunner(sc).run().start_driver()

```

