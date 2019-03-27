from ray_poc.rayrunner import RayRunner
from ray_poc.util.spark import init_spark_on_yarn


# def init_spark_on_yarn(spark_home="/home/zhichao/god/spark-2.4.0-bin-hadoop2.7/",java_home="/home/zhichao/god/jdk1.8.0_101/",hadoop_conf="/home/zhichao/god/yarn55_config/etc/hadoop",
#                        spark_yarn_jars="hdfs://172.168.2.181:9000/zhichao/ray/spark-2.4-jar.zip",
#                        python_env_archive="/home/zhichao/god/ray_35.tar.gz",
#                        python_zip_file="/home/zhichao/god/analytics-zoo/pyzoo/ray_poc.zip",
#                        num_executor=3,
#                        executor_cores=28,
#                        executor_memory="100g",
#                        driver_memory="10g",
#                        driver_cores=10,
#                        spark_executor_pyspark_memory="60g",
#                        hadoop_user_name="root"):
spark_home = "/home/zhichao/god/spark-2.4.0-bin-hadoop2.7/"
java_home = "/home/zhichao/god/jdk1.8.0_101/"
hadoop_conf = "/home/zhichao/god/yarn55_config/etc/hadoop",
spark_yarn_jars = "hdfs://172.168.2.181:9000/zhichao/ray/spark-2.4-jar.zip",
python_env_archive = "/home/zhichao/god/ray_35.tar.gz",
python_zip_file = "/home/zhichao/god/analytics-zoo/pyzoo/ray_poc.zip"

sc, python_loc = init_spark_on_yarn()

rayRunner = RayRunner(sc, python_loc)

redis_address = rayRunner.run()
# redis_address is 56 but it returnning 59

print(redis_address)

# TODO: setting the python log? or just print or redirect to elsewhere?
