import os
import sys

SPARK_HOME = "/home/lizhichao/bin/spark-2.4.0-bin-hadoop2.7/"
JAVA_HOME = "/home/lizhichao/bin/jdk1.8.0_101/"
os.environ[
    "JAVA_HOME"] = JAVA_HOME  # this is a must otherwise javagateway would throw cannot connet error
os.environ["SPARK_HOME"] = SPARK_HOME
# os.environ["HADOOP_CONF_DIR"] = "/opt/work/hadoop_conf/hadoop"
# os.environ['PYSPARK_SUBMIT_ARGS'] = '--master yarn --archives /opt/work/py27.tar.gz#python_env --executor-cores 20 --executor-memory 50g --num-executors 2 pyspark-shell'
# os.environ['PYSPARK_PYTHON'] = "python_env/bin/python"

os.environ['PYSPARK_PYTHON'] = "/home/lizhichao/anaconda3/envs/py27/bin/python"

sys.path.append("%s/python/lib/pyspark.zip" % SPARK_HOME)
sys.path.append("%s/python/lib/py4j-0.10.7-src.zip" % SPARK_HOME)

import pyspark
import time

print(pyspark.__file__)
from pyspark import *
from pyspark.sql import SparkSession
from zoo.common.ray_poc.util.safe_shell_exec import get_ip_address, simple_execute
from zoo.common.ray_poc.util.safe_shell_exec import execute


def ray_poc():

    # TypeError: __init__() got an unexpected keyword argument 'auth_token' <- pip install pyspark==2.4.0 solved.

    # spark_conf = SparkConf().setMaster("local[4]").set("spark.driver.memory", "2g")
    spark = SparkSession.builder.master("local[4]").config(key="spark.driver.memory", value="4g").getOrCreate()
    sc = spark.sparkContext
    master_file = "master.lock"

# ip + partition is the id
#

    # def _mapper(index, iter):
    #     tc = BarrierTaskContext.get()
    #     if tc.partitionId() == 0:
    #         print("master")
    #         tc = BarrierTaskContext.get()
    #         tc.master = "true"
    #         print(tc)
    #         yield tc
    #     else:
    #         tc = BarrierTaskContext.get()
    #         print(tc.master)
    #         yield 1




    def gather_ip(iter):
        yield get_ip_address()
        # tc = BarrierTaskContext.get()
        # if tc.partitionId() == 0:
        #     print("master")
        #     return [i.address for i in tc.getTaskInfos()]
        # else:
        #     return []

    def get_ray_command_path():
        python_loc = os.environ['PYSPARK_PYTHON']
        return "{}/ray".format("/".join(python_loc.split("/")[:-1]))

    def start_ray_service(iter):
        # TODO: how can we get the ip:port within a task?
        tc = BarrierTaskContext.get()
        # The address is sorted by partitionId according to the comments
        # Partition 0 is the Master
        task_addrs = [taskInfo.address for taskInfo in tc.getTaskInfos()]
        print("$$$$$$")
        print(task_addrs)
        master_ip = task_addrs[0].split(":")[0]

        print("current_address {}".format(task_addrs[tc.partitionId()]))
        print("master_address {}".format(master_ip))

        # master_ip = tc.getLocalProperty("redis_ip")
        # current_ip = get_ip_address()
        RAY_COMMAND = get_ray_command_path()

        # clean the env first
        simple_execute("{} stop".format(RAY_COMMAND))
        time.sleep(5)

        def start_raylet():
            redis_address = "{}:{}".format(master_ip, REDIS_PORT)
            command = "nohup {} start --redis-address {} --redis-password 123456".format(RAY_COMMAND, redis_address)
            print(command)
            simple_execute(command)

        if tc.partitionId() == 0:
            print("working dir: {}".format(os.getcwd()))
            command = "nohup {} start --head --redis-port {} --redis-password 123456".format(RAY_COMMAND, REDIS_PORT)
            print(command)
            # TODO redis port should be randomly searched
            simple_execute(command)
            time.sleep(5)
            yield task_addrs[0]
        else:
            print("partition id is : {}".format(tc.partitionId()))
            start_raylet()
            time.sleep(5)



    NUM_WORKERS = 1
    REDIS_PORT = "5347"
    # one ray.master two ray.worker
    # ips = sc.range(0, NUM_WORKERS + 1, numSlices = NUM_WORKERS + 1).barrier().mapPartitions(gather_ip).collect()
    # ips = [i.split(":")[0] for i in addresses]
    # sc.setLocalProperty("redis_ip", ips[0])
    # sc.setLocalProperty("redis_port", REDIS_PORT)

    redis_host_address = sc.range(0, NUM_WORKERS + 1, numSlices = NUM_WORKERS + 1).barrier().mapPartitions(start_ray_service).collect()
    assert len(redis_host_address) == 1, "we should only create 1 redis"
    redis_host_address = redis_host_address[0]

    # procs = sc.range(0, 1).barrier().mapPartitions(start_master).map(lambda _: )
    # result = procs.mapPartitionsWithIndex(start_master).collect()
    # ips

    # Create a driver and execute something here.

    def start_driver_service(iter):
        tc = BarrierTaskContext.get()
        RAY_COMMAND = get_ray_command_path()
        task_addrs = [taskInfo.address for taskInfo in tc.getTaskInfos()]
        if task_addrs[tc.partitionId()] == redis_host_address:
            # we are at the master executor
            print("working dir: {}".format(os.getcwd()))
            print("Starting the driver process for ray")

            import ray
            print("###redis address: " + redis_host_address.split(":")[0])
            ray.init("{}:{}".format((redis_host_address.split(":")[0], REDIS_PORT)))

            @ray.remote
            def remote_function():
                return 1

            result = [ray.get(remote_function.remote()) for i in range(0, NUM_WORKERS)]
            yield result


    result = sc.range(0, NUM_WORKERS + 1, numSlices = NUM_WORKERS + 1).barrier().mapPartitions(start_driver_service).collect()

    result




    # TODO: register cleaner for each partition
    # def cleaner(_):
    #     simple_execute("{} stop".format(get_ray_command_path()))
    # sc.range(0, NUM_WORKERS + 1, numSlices=NUM_WORKERS + 1).barrier().mapPartitions(
    #     cleaner).collect()
ray_poc()




