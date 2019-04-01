import os
import time
import threading
from pyspark import BarrierTaskContext
import atexit
import re
import signal
import sys
import itertools


from ray_poc.util.process import session_execute, ProcessResource


def _gen_shutdown_per_node(pgids):
    def _shutdown_per_node(iter):
        print("shutting down pgid: {}".format(pgids))
        for pgid in pgids:
            print("killing {}".format(pgid))
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                print("WARNING: cannot find pgid: {}".format(pgid))

    return _shutdown_per_node

class RayContext(object):
    """
    This should be a pickable class.
    """

    def get_MKL_config(self, cores):
       return {"intra_op_parallelism_threads":cores,
       "inter_op_parallelism_threads":cores,
       "OMP_NUM_THREADS":cores,
       "KMP_BLOCKTIME":0,
       "KMP_AFFINITY":"granularity = fine, verbose, compact, 1, 0",
       "KMP_SETTINGS":0
               }  # MALLOC_ARENA_MAX

    def prepare_env(self, cores):
        modified_env = os.environ.copy()
        cwd = os.getcwd()
        modified_env["PATH"] = "{}/{}:{}".format(cwd, "/".join(self.python_loc.split("/")[:-1]), os.environ["PATH"])
        modified_env.update(self.get_MKL_config(cores))
        print("The command searching path is: {}".format(modified_env["PATH"]))


    def __init__(self, python_loc, redis_port, slave_cores, master_cores, redis_max_memory, password):
        self.python_loc = python_loc
        self.redis_port = redis_port
        self.password = password
        self.slave_cores = slave_cores
        self.master_cores = master_cores
        self.redis_max_memory=redis_max_memory
        self.ray_exec = self.get_ray_exec()
        self.WAITING_TIME_SEC = 5

    def start_master(self):
        """
        Start the Master for Ray
        :return:
        """

        # command = "{} start --block --head --redis-port {} --redis-password {} --num-cpus {} --redis-max-memory {}".format(
        #     self.ray_exec, self.redis_port, self.password,
        #     self.master_cores,
        # self.redis_max_memory)
        modified_env = self.prepare_env(self.master_cores)
        command = "{} start --head --redis-port {} --redis-password {} --num-cpus {}".format(
            self.ray_exec, self.redis_port, self.password,
            self.master_cores)

        print("Starting ray master by running {}".format(command))
        try:
            session_execute(command, env=modified_env)
        except Exception as e:
            ProcessResource.exception.append(e)

        time.sleep(self.WAITING_TIME_SEC)

    def start_raylet(self, redis_address):
        """
        Start the Slave for Ray
        :return:
        """
        # command = "{} start --block --redis-address {} --redis-password  {} --num-cpus {} --redis-max-memory {}".format(
        #     self.ray_exec, redis_address, self.password, self.slave_cores, self.redis_max_memory)
        command = "{} start aa --redis-address {} --redis-password  {} --num-cpus {}".format(
            self.ray_exec, redis_address, self.password, self.slave_cores)
        print("".format(command))

        modified_env = self.prepare_env(self.master_cores)
        try:
            session_execute(command, env=modified_env)
        except Exception as e:
            ProcessResource.exception.append(e)

        time.sleep(self.WAITING_TIME_SEC)

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
            self.stop_ray()
            redis_address = "{}:{}".format(master_ip, self.redis_port)
            if tc.partitionId() == 0:
                print("partition id is : {}".format(tc.partitionId()))
                self.start_master()
                yield redis_address, ProcessResource.pgids, ProcessResource.exception
            else:
                print("partition id is : {}".format(tc.partitionId()))
                self.start_raylet(redis_address=redis_address)
                yield None, ProcessResource.pgids, ProcessResource.exception
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
                                      slave_cores=self.executor_cores,
                                      master_cores=self.executor_cores,
                                      redis_max_memory=self.redis_max_memory,
                                      password=password)

    def get_executor_cores(self):
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

    def _parse_running_meta(self, running_meta):
        addrs = []
        pgids = []
        exceptions = []
        for pmeta in running_meta:
            addrs.append(pmeta[0])
            pgids.append(pmeta[1])
            exceptions.append(pmeta[2])
        addrs = [addr for addr in addrs if addr]
        pgids=list(itertools.chain.from_iterable(pgids))
        exceptions=list(itertools.chain.from_iterable(exceptions))
        assert len(addrs) == 1, "we should only have one redis address"
        return addrs[0], pgids, exceptions

    def _run(self):
        # TODO: it should return [IP, gpid], then we can only kill the gpid if the ip address match.
        running_meta = self.sc.range(0,
                                      self.num_executors,
                                      numSlices=self.num_executors).barrier().mapPartitions(
            self.ray_context.gen_ray_booter()).collect()
        redis_address, pgids, exceptions = self._parse_running_meta(running_meta)

        self.redis_address = redis_address
        ProcessResource.pgids = pgids
        # what if half of the services is started and then exception happen?
        self.register_shutdown_hook()
        if len(exceptions) > 0:
            raise Exception(exceptions)
        return redis_address


    def register_shutdown_hook(self):
        def _shutdown():
            self.sc.range(0,
                          self.num_executors,
                          numSlices=self.num_executors).map(_gen_shutdown_per_node(ProcessResource.pgids)).collect()

        def _signal_shutdown(_signo, _stack_frame):
            self._shutdown()
            sys.exit(0)

        # TODO: are there any other signal we want to handle?
        atexit.register(_shutdown)
        signal.signal(signal.SIGTERM, _signal_shutdown)
        signal.signal(signal.SIGINT, _signal_shutdown)


    # def stop(self):
    #     self.sc.range(0,
    #                   self.num_executors + 1,
    #                   numSlices=self.num_executors + 1).barrier().mapPartitions(
    #         self.ray_context.stop_ray()).collect()

    def start_driver(self):
        """
        It would create a driver and a dummy local raylet
        :return:
        """
        import ray
        ray.init(redis_address=self.redis_address,
                 redis_password=self.password)

        # catch all signal.
        # for i in [x for x in dir(signal) if x.startswith("SIG")]:
        #     try:
        #         signum = getattr(signal, i)
        #         signal.signal(signum, sighandler)
        #     except (OSError, RuntimeError) as m:  # OSError for Python3, RuntimeError for 2
        #         print ("Skipping {}".format(i))
