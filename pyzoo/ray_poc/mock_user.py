

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import ray
from ray.experimental.sgd.sgd import DistributedSGD
from ray.experimental.sgd.tfbench.test_model import TFBenchModel

from ray_poc.util.process import simple_execute


# <class 'list'>: ['172.168.2.161:5346']
# need to reduce the plasma memory consumption as well
def start_dummy_ray_worker(redis_address, redis_password):
    num_cores = 4
    command = "nohup {} start --redis-address {} --redis-password  {} --num-cpus {} --object-store-memory {}".format(
        "ray", redis_address, redis_password, num_cores, "1000000000")
    print("".format(command))
    simple_execute(command)

# devices_per_worker: cpu_nums
# all_reduce_alg: strategy for gradient sync within the same worker
def test_sgd(batch_size=2, num_workers=4, devices_per_worker=4, strategy="ps", warmup=True, num_iters=10, stats_interval=2):
    model_creator = (
        lambda worker_idx, device_idx: TFBenchModel(
            batch=batch_size, use_cpus=True))

    sgd = DistributedSGD(
        model_creator,
        num_workers=num_workers,
        devices_per_worker=devices_per_worker,
        gpu=False,
        strategy=strategy,
        # grad_shard_bytes=0,
        grad_shard_bytes=10000000,
        all_reduce_alg='simple')

    if warmup:
        sgd.warmup()

    t = []

    for i in range(num_iters):
        start = time.time()
        fetch_stats = i % stats_interval == 0
        print("== Step {} ==".format(i))
        stats = sgd.step(fetch_stats=fetch_stats)
        # batch_size is per device(core)
        ips = ((batch_size * num_workers * devices_per_worker) /
               (time.time() - start))
        print("Iteration time", time.time() - start, "Images per second", ips)
        t.append(ips)
        if fetch_stats:
            print("Current loss", stats)

    print("Peak throughput", max(sum(t[i:i + 5]) / 5 for i in range(len(t))))

def start_ray_driver(redis_address='172.168.2.159:5346',
    redis_password='123456', object_store_memory="2g"):

    # start_dummy_ray_worker(redis_address, redis_password)
    # TODO: we need to wait for worker ready here....
    ray.shutdown()
    ray.init(redis_address=redis_address,
             redis_password=redis_password)

# start_ray_driver(redis_address="172.168.2.156:5346")
start_ray_driver(redis_address="172.168.2.118:8234")

test_sgd()

ray.shutdown()