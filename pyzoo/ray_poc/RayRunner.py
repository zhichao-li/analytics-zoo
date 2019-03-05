import os
import time
from pyspark import BarrierTaskContext
# builder pattern for python ?

class RayRunner(object):
    # TODO: redis_port should be retrieved by random searched
    def __init__(self, sc, python_loc, redis_port="5346", password="123456"):
        self.sc = sc
        self.spark_runtime_conf = sc._conf
        self.python_loc = python_loc
        self.redis_port = redis_port
        self.password = password

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['sc']
        return state

    # how many executors?

    def get_num_executors(self):
        return int(self.spark_runtime_conf.get("spark.executor.instances"))

    def run(self):
        num_executors = self.get_num_executors()
        redis_address = self.sc.range(0, num_executors + 1,
                              numSlices=num_executors + 1).barrier().mapPartitions(self._start_ray_services).collect()
        return redis_address


    def start_master(self, redis_port):
        """
        Start the Master for Ray
        :return:
        """
        command = "nohup {} start --head --redis-port {} --redis-password {} --num-cpus {}".format(self.get_ray_exec(), redis_port, self.password, self.spark_runtime_conf.get("spark.executor.cores"))
        print("Starting ray master by running {}".format(command))

    def start_raylet(self, redis_address):
        """
        Start the Slave for Ray
        :return:
        """
        command = "nohup {} start --redis-address {} --redis-password  {} --num-cpus {}".format(self.get_ray_exec(), redis_address, self.password, self.spark_runtime_conf.get("spark.executor.cores"))
        print("".format(command))
        RayRunner.simple_execute(command)

    def stop_ray(self):
        RayRunner.simple_execute("{} stop".format(self.get_ray_exec()))

    def start_driver(self):
        """
        It would create a driver and a dummy local raylet
        :return:
        """

    def get_ray_exec(self):
        python_bin_dir = "/".join(self.python_loc.split("/")[:-1])
        return "{}/python {}/ray".format(python_bin_dir, python_bin_dir)

    def _start_ray_services(self, iter):
        tc = BarrierTaskContext.get()
        # The address is sorted by partitionId according to the comments
        # Partition 0 is the Master
        task_addrs = [taskInfo.address for taskInfo in tc.getTaskInfos()]
        print(task_addrs)
        master_ip = task_addrs[0].split(":")[0]
        print("current address {}".format(task_addrs[tc.partitionId()]))
        print("master address {}".format(master_ip))
        # clean the env first
        self.stop_ray()
        redis_address = "{}:{}".format(master_ip, self.redis_port)
        if tc.partitionId() == 0:
            self.start_master(redis_port=self.redis_port)
            yield redis_address
        else:
            print("partition id is : {}".format(tc.partitionId()))
            self.start_raylet(redis_address=redis_address)
        tc.barrier()

    @staticmethod
    def simple_execute(command, env=None, stdout=None, stderr=None):
        import subprocess
        process = subprocess.Popen(
            command,
            shell=True,
            env=env,
            cwd=None,
            stdout=stdout,
            stderr=stderr)

#     @staticmethod
#     def get_resource_from_spark(sc):
#         master = sc._conf.get("spark.master")
#         if "yarn" in master.lower():
#             spark_executor_memory = sc._conf.get("spark.executor.memory")
#             spark_executor_cores = sc._conf.get("spark.executor.cores")
#             spark_driver_memory = sc._conf.get("spark.driver.memory")
#             return  SparkRuntimeConfig(spark_driver_memory=spark_driver_memory,
#                                        spark_executor_memory=spark_executor_memory,
#                                        spark_executor_cores=spark_executor_cores)
#         else:
#             raise NotImplementedError("Not supported type: {}".format(master))
#
#
# class SparkRuntimeConfig(object):
#     def __init__(self, spark_driver_memory, spark_executor_memory, spark_executor_cores):
#         self.spark_driver_memory = spark_driver_memory
#         self.spark_executor_memory = spark_executor_memory
#         self.spark_executor_cores = spark_executor_cores

