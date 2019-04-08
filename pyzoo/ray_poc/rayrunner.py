import os
import time
import threading
from pyspark import BarrierTaskContext
import atexit
import re
import signal
import sys
import itertools

from ray_poc.util import is_local
from ray_poc.util.process import session_execute, ProcessMonitor


class RayContext(object):
    """
    This should be a pickable class.
    """

    def get_MKL_config(self, cores):
       return {"intra_op_parallelism_threads":str(cores),
       "inter_op_parallelism_threads":str(cores),
       "OMP_NUM_THREADS":str(cores),
       "KMP_BLOCKTIME":"0",
       "KMP_AFFINITY":"granularity = fine, verbose, compact, 1, 0",
       "KMP_SETTINGS":"0"
               }

    def prepare_env(self, cores=None):
        modified_env = os.environ.copy()
        cwd = os.getcwd()
        modified_env["PATH"] = "{}/{}:{}".format(cwd, "/".join(self.python_loc.split("/")[:-1]), os.environ["PATH"])
        modified_env.pop("MALLOC_ARENA_MAX", None)
        modified_env.pop("RAY_BACKEND_LOG_LEVEL", None)
        if cores:
            modified_env.update(self.get_MKL_config(cores))
            print(modified_env)
        print("The command searching path is: {}".format(modified_env["PATH"]))
        return modified_env


    def __init__(self, python_loc, redis_port, mkl_cores, redis_max_memory, password):
        self.python_loc = python_loc
        self.redis_port = redis_port
        self.password = password
        self.mkl_cores=mkl_cores
        self.redis_max_memory=redis_max_memory
        self.ray_exec = self.get_ray_exec()
        self.WAITING_TIME_SEC = 10

    def _get_ray_cores(self):
        # 2 for worker 1 for parameterserver.
       return 3



    def start_master(self):
        """
        Start the Master for Ray
        :return:
        """
        modified_env = self.prepare_env(self.mkl_cores)
        command = "{} start --head --include-webui --redis-port {} --redis-password {} --num-cpus {} ".format(
            self.ray_exec, self.redis_port, self.password,
            self._get_ray_cores())
        print("Starting ray master by running: {}".format(command))
        process_info = session_execute(command, env=modified_env, tag="ray_master")
        # TODO: we need to think serious about the time setting otherwise client would not be able to connect to the master
        time.sleep(self.WAITING_TIME_SEC)
        return process_info

    def start_raylet(self, redis_address):
        """
        Start the Slave for Ray
        :return:
        """
        command = "{} start --redis-address {} --redis-password  {} --num-cpus {} ".format(
            self.ray_exec, redis_address, self.password, self._get_ray_cores())
        print("Starting raylet by running: {}".format(command))

        modified_env = self.prepare_env(self.mkl_cores)
        time.sleep(self.WAITING_TIME_SEC)
        return session_execute(command, env=modified_env, tag="raylet")


    def stop_ray(self):
        # it would have issue if two ray process running on the same node.
        pass
        # command="{} stop".format(self.ray_exec)
        # print("cleaning the ray cluster: {}".format(command))
        # modified_env = self.prepare_env(self.master_cores)
        # session_execute(command, env=modified_env)
        # time.sleep(self.WAITING_TIME_SEC)

    def get_ray_exec(self):
        python_bin_dir = "/".join(self.python_loc.split("/")[:-1])
        return "{}/python {}/ray".format(python_bin_dir, python_bin_dir)

    def gen_ray_booter(self):
        def _start_ray_services(iter):
            tc = BarrierTaskContext.get()
            # The address is sorted by partitionId according to the comments
            # Partition 0 is the Master
            task_addrs = [taskInfo.address for taskInfo in tc.getTaskInfos()]
            print(task_addrs)
            master_ip = task_addrs[0].split(":")[0]
            print("current address {}".format(task_addrs[tc.partitionId()]))
            print("master address {}".format(master_ip))
            # clean the env first
            # cannot allocate two task to the same node as ray stop would grep and kill all of the ray related process
            # spark would try to spread task to different machine, but they still be possible that they share the same node.
            # this must be fixed !!
            # we can walk around this by increase the vcores per executor for now
            redis_address = "{}:{}".format(master_ip, self.redis_port)
            if tc.partitionId() == 0:
                print("partition id is : {}".format(tc.partitionId()))
                process_info = self.start_master()
                process_info.master_addr = redis_address
                yield process_info
            else:
                print("partition id is : {}".format(tc.partitionId()))
                process_info = self.start_raylet(redis_address=redis_address)
                yield process_info
            tc.barrier()

        return _start_ray_services


class RayRunner(object):
    # TODO: redis_port should be retrieved by random searched
    def __init__(self, sc, python_loc, redis_port="5346", password="123456"):
        self.sc = sc
        self.executor_cores = self.get_executor_cores()
        self.num_executors = self.get_num_executors()
        self.redis_max_memory = self.sc._conf.get("spark.executor.pyspark.memory")
        # assert self.redis_max_memory, "you should set spark.executor.pyspark.memory"
        self.ray_context = RayContext(
                                      python_loc=python_loc,
                                      redis_port=redis_port,
                                      mkl_cores=self._get_mkl_cores(),
                                      redis_max_memory=self.redis_max_memory,
                                      password=password)

    def _get_mkl_cores(self):
        if is_local(self.sc):
            return 1  # TODO: make this configurable
        else:
            return self.spark_executor_cores

    def get_executor_cores(self):
        # TODO: if cores > 1, then there would be possible that 2 sgd_worker run on the same node
        if "local" in self.sc.master:
            return int(re.match(r"local\[(.*)\]", self.sc.master).group(1))
        else:
            return self.sc._conf.get("spark.executor.cores")


    def get_num_executors(self):
        if "local" in self.sc.master:
            return 2
        else:
            return int(self.sc._conf.get("spark.executor.instances"))

    def run(self):
        # TODO: before involving thread, we need to figure out how to return the redis_address to user

        # ray_launching_thread = threading.Thread(target=self._run, args=(self.result_queue, ))
        # ray_launching_thread.start()
        # redis_address = self.result_queue.get()
        # return redis_address
        return self._run()


    def _run(self):
        # TODO: it should return [IP, gpid], then we can only kill the gpid if the ip address match.
        ray_rdd = self.sc.range(0,
                                      self.num_executors,
                                      numSlices=self.num_executors)
        process_infos = ray_rdd.barrier().mapPartitions(
            self.ray_context.gen_ray_booter()).collect()

        processMonitor = ProcessMonitor(process_infos, self.sc, ray_rdd)
        self.redis_address = processMonitor.master.master_addr
        return self.redis_address



    def purge_ray_processes(self):
        def _gen_purge(_):
            # TODO: print some ip info back?
            modified_env = self.prepare_env(self.master_cores)
            command = "{} stop".format(self.ray_exec)
            session_execute(command, env=modified_env)

        def purge(_):
            self.sc.range(0,
                          self.num_executors,
                          numSlices=self.num_executors).barrier().mapPartitions(
                _gen_purge).collect()

    def start_driver(self):
        """
        It would create a driver and a dummy local raylet
        :return:
        """
        import ray
        ray.init(redis_address=self.redis_address,
                 redis_password=self.password)


