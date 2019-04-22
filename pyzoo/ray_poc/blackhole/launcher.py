import os
import sys

from ray_poc.rayrunner import RayRunner

SPARK_HOME = "/home/zhichao/god/spark-2.4.0-bin-hadoop2.7/"
# SPARK_HOME = "/home/zhichao/god/zhichao/spark-2.2.0-bin-hadoop2.7/"
JAVA_HOME = "/home/zhichao/god/jdk1.8.0_101/"
os.environ[
    "JAVA_HOME"] = JAVA_HOME  # this is a must otherwise javagateway would throw cannot connet error
os.environ["SPARK_HOME"] = SPARK_HOME
os.environ["HADOOP_CONF_DIR"] = "/home/zhichao/god/yarn55_config/etc/hadoop"
os.environ['PYSPARK_SUBMIT_ARGS'] = '--master yarn --archives /home/zhichao/god/ray_35.tar.gz#python_env --py-files /home/zhichao/god/analytics-zoo/pyzoo/ray_poc.zip --executor-cores 10 --executor-memory 50g --num-executors 2 pyspark-shell'
os.environ['PYSPARK_PYTHON'] = "python_env/bin/python"
os.environ['HADOOP_USER_NAME'] = "root"

sys.path.append("%s/python/lib/pyspark.zip" % SPARK_HOME)
sys.path.append("%s/python/lib/py4j-0.10.7-src.zip" % SPARK_HOME) #spark2.4.0
# sys.path.append("%s/python/lib/py4j-0.10.4src.zip" % SPARK_HOME) #spark2.2.0

import pyspark
import time

print(pyspark.__file__)
from pyspark import *
from pyspark.sql import SparkSession
import ray

# python_loc would be ser to remote along with function
python_loc = os.environ['PYSPARK_PYTHON']

# sc._conf.get("spark.driver.memory"
# TypeError: __init__() got an unexpected keyword argument 'auth_token' <- pip install pyspark==2.4.0 solved.
# spark_conf = SparkConf().setMaster("local[4]").set("spark.driver.memory", "2g")
spark = SparkSession.builder.config(key="spark.driver.memory", value="4g").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("DEBUG")

value = sc.range(0, 3, numSlices=3).collect()
print(value)

@ray.remote
def remote_aaa():
    return 1

def start_driver_code():
    print("before init")
    redis_addr = "10.239.10.105:5347"
    import ray
    ray.init(redis_address=redis_addr,
             redis_password="123456")
    print("after init")

    result = remote_aaa.remote()
    # time.sleep(100)

    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("hi" + str(ray.get(result)))
    try:
        ray.put("I'm here")
    except :
        print("bad boy")
    finally:
        print("finally here")

    import sys
    sys.stdout.flush()

    time.sleep(100000)
    # print(result)

# def driver_func(_):
#     start_driver_code()
#     master_file = "master.created"
#     if (not os.path.isfile(master_file)):
#         print("creating master file")
#         with open(os.path.join(os.getcwd(), master_file), 'w') as f:
#             f.write('master is created.')
#             f.close()
#         print("create master file")
#         start_driver_code()
#
#         yield []
#     else:
#         print("empty action")
#         yield []

    # time.sleep(1000000)




def ray_poc():


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
        python_bin_dir = "/".join(python_loc.split("/")[:-1])
        return "{}/python {}/ray".format(python_bin_dir, python_bin_dir)


    # Create a driver and execute something here.

    # def start_driver_service():
    #
    #         yield result

    def start_raylet(ray_exec, redis_addr):
        command = "nohup {} start --redis-address {} --redis-password 123456".format(ray_exec, redis_addr)
        print(command)
        RayRunner.simple_execute(command)

    # def start_ray_service(index, iter):
    #
    #     print("current_address {}".format(get_ip_address()))
    #
    #     current_ip = get_ip_address()
    #     ray_exec = get_ray_command_path()
    #
    #     # clean the env first
    #     simple_execute("{} stop".format(ray_exec))
    #     time.sleep(5)
    #
    #     redis_address = "{}:{}".format(master_ip, REDIS_PORT)
    #
    #     if tc.partitionId() == 0:
    #         print("working dir: {}".format(os.getcwd()))
    #         command = "nohup {} start --head --redis-port {} --redis-password 123456".format(RAY_COMMAND, REDIS_PORT)
    #         print(command)
    #         # TODO redis port should be randomly searched
    #         simple_execute(command)
    #         time.sleep(5)
    #         # yield task_addrs[0]
    #     else:
    #         print("partition id is : {}".format(tc.partitionId()))
    #         start_raylet()
    #         time.sleep(5)
    #
    #     if index == 0:



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
        RayRunner.simple_execute("{} stop".format(RAY_COMMAND))
        time.sleep(5)

        def start_raylet():
            redis_address = "{}:{}".format(master_ip, REDIS_PORT)
            command = "nohup {} start --redis-address {} --redis-password 123456".format(RAY_COMMAND, redis_address)
            print(command)
            RayRunner.simple_execute(command)

        if tc.partitionId() == 0:
            print("working dir: {}".format(os.getcwd()))
            command = "nohup {} start --head --redis-port {} --redis-password 123456".format(RAY_COMMAND, REDIS_PORT)
            print(command)
            # TODO redis port should be randomly searched
            RayRunner.simple_execute(command)
            time.sleep(5)
            yield task_addrs[0]
        else:
            print("partition id is : {}".format(tc.partitionId()))
            start_raylet()
            time.sleep(5)

        tc.barrier()

        # tc = BarrierTaskContext.get()
        # RAY_COMMAND = get_ray_command_path()
        # task_addrs = [taskInfo.address for taskInfo in tc.getTaskInfos()]
        # print("driver_service")
        # print(task_addrs)
        # if tc.partitionId() == 0:
        #     # we are at the master executor
        #     print("working dir: {}".format(os.getcwd()))
        #     print("Starting the driver process for ray")
        #     # driver_func(master_ip, REDIS_PORT)
        #     import ray
        #     ray.init(redis_address="{}:{}".format(master_ip, REDIS_PORT),
        #              redis_password="123456")
        #     print("after init")
        #
        #     driver_func()
        #     print("end of driver")
        #     yield []
        #
        #
        # tc.barrier()

        yield []




    NUM_WORKERS = 1
    REDIS_PORT = "5347"
    # one ray.master two ray.worker
    # ips = sc.range(0, NUM_WORKERS + 1, numSlices = NUM_WORKERS + 1).barrier().mapPartitions(gather_ip).collect()
    # ips = [i.split(":")[0] for i in addresses]
    # sc.setLocalProperty("redis_ip", ips[0])
    # sc.setLocalProperty("redis_port", REDIS_PORT)

    redis_host = sc.range(0, NUM_WORKERS + 1, numSlices = NUM_WORKERS + 1).barrier().mapPartitions(start_ray_service).collect()
    redis_host
    # assert len(redis_host_address) == 1, "we should only create 1 redis"
    # redis_host_address = redis_host_address[0]

    # procs = sc.range(0, 1).barrier().mapPartitions(start_master).map(lambda _: )
    # result = procs.mapPartitionsWithIndex(start_master).collect()
    # ips




    # result = sc.range(0, NUM_WORKERS + 1, numSlices = NUM_WORKERS + 1).barrier().mapPartitions(driver_func).collect()

    # result

    # start_driver_code()




    # TODO: register cleaner for each partition
    # def cleaner(_):
    #     simple_execute("{} stop".format(get_ray_command_path()))
    # sc.range(0, NUM_WORKERS + 1, numSlices=NUM_WORKERS + 1).barrier().mapPartitions(
    #     cleaner).collect()
ray_poc()

print("hello yarn")



