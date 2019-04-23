from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import ray
from ray.experimental.sgd.sgd import DistributedSGD
from ray.experimental.sgd.tfbench.test_model import TFBenchModel

# <class 'list'>: ['172.168.2.161:5346']
# need to reduce the plasma memory consumption as well
from zoo.ray.util.process import session_execute




# devices_per_worker: cpu_nums
# all_reduce_alg: strategy for gradient sync within the same worker
def test_sgd(batch_size, num_workers, grad_shard_bytes, devices_per_worker=2, strategy="ps", warmup=True, num_iters=50, stats_interval=2, ):
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
        grad_shard_bytes=grad_shard_bytes,
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

