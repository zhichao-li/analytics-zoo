from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import random
import time

import numpy as np

import ray

from ray_poc.allreduce.ps import ShardedParameterServer

logger = logging.getLogger(__name__)

class RayDataSet(object):
    pass

class DummyRayDataSet(RayDataSet):
    def __init__(self, shape):
        self.shape = shape

    def next_batch(self):
        return np.random.rand(self.shape)

class ModelWorker(object):

    def __init__(self, model, ray_data_set, num_workers):
        self.model = model
        self.num_workers = num_workers

    def compute_next(self, *parameters):
        """
        It would return a sharded grads here.
        Each parameter should be a 1-D vector
        """
        # concate grads
        # TODO:
        parameters = np.concatenate(parameters)
        # TODO: model should have api: set_flat_paramters
        self.model.set_flat_parameters(parameters)
        # TODO: model should have api for getting next batch
        input = self.ray_data_set()
        grads = self.model.compute(input)
        # split grads here
        # TODO: enhance the split method here as it would raise exception if cannot be evently divided.
        sharded_grads = np.split(grads, self.num_workers)
        return sharded_grads



class DistributedOptimizer(object):

    def __init__(self,
                 model,
                 num_workers):
        self.num_workers = num_workers
        requests = {"num_cpus": 1}

        ModelWorkerWithResource = ray.remote(**requests)(ModelWorker)

        # TODO: remove this fake weights
        weights = np.ones(64000000)
        sharded_weights = np.split(weights, self.num_workers)
        # TODO: implement get_grads
        #grads = model.get_flat_grads()
        grads = np.ones(64000000) # 64M
        sharded_grads = np.split(grads, self.num_workers)

        sharded_weight_ids = [ray.put(w) for w in sharded_weights]

        self.workers = []
        self.pss = []
        logger.info(
            "Creating parameter server ({} total)".format(
                num_workers))
        for ps_index in range(num_workers):
            self.pss.append(
                ShardedParameterServer.remote(sharded_weight_ids[ps_index]))

        logger.info(
            "Creating model workers ({} total)".format(
                num_workers))
        for worker_index in range(num_workers):
            self.workers.append(
                ModelWorkerWithResource.remote(model, num_workers))

        steps = 10
        for step in steps:
            self.run_step()

    def run_step(self):
        # workers of sharded_grads
        sharded_grad_ids = []
        # pull weights from ps
        for worker in self.workers:
            # 1) pull the latest weights from ps
            parameter_ids = [ps.get_parameters.remote() for ps in self.pss]
            # 2) compute the grads
            sharded_grad_ids.append(worker.compute_next(parameter_ids))

        grads_per_ps = zip(*sharded_grad_ids)
        assert len(grads_per_ps[0]) == self.num_workers, "we should get correct grads for each ps"
        # 3) push and aggregate grads on ps
        for index, grads in enumerate(grads_per_ps):
            self.pss[index].aggregate_grads_update_weights(grads)


def _distributed_sgd_step(actors, ps_list, fetch_stats, write_timeline):
    # Preallocate object ids that actors will write gradient shards to
    grad_shard_oids_list = [[np.random.bytes(20) for _ in ps_list]
                            for _ in actors]
    logger.debug("Generated grad oids")

    # Preallocate object ids that param servers will write new weights to
    accum_shard_ids = [np.random.bytes(20) for _ in ps_list]
    logger.debug("Generated accum oids")

    # Kick off the fused compute grad / update weights tf run for each actor
    losses = []
    for actor, grad_shard_oids in zip(actors, grad_shard_oids_list):
        losses.append(
            actor.ps_compute_apply.remote(
                grad_shard_oids,
                accum_shard_ids,
                write_timeline=write_timeline))
    logger.debug("Launched all ps_compute_applys on all actors")

    # Issue prefetch ops
    for j, (ps, weight_shard_oid) in list(
            enumerate(zip(ps_list, accum_shard_ids)))[::-1]:
        to_fetch = []
        for grad_shard_oids in grad_shard_oids_list:
            to_fetch.append(grad_shard_oids[j])
        random.shuffle(to_fetch)
        ps.prefetch.remote(to_fetch)
    logger.debug("Launched all prefetch ops")

    # Aggregate the gradients produced by the actors. These operations
    # run concurrently with the actor methods above.
    ps_gets = []
    for j, (ps, weight_shard_oid) in list(
            enumerate(zip(ps_list, accum_shard_ids)))[::-1]:
        ps.add_spinwait.remote([gs[j] for gs in grad_shard_oids_list])
        ps_gets.append(ps.get.remote(weight_shard_oid))
    logger.debug("Launched all aggregate ops")

    if write_timeline:
        timelines = [ps.get_timeline.remote() for ps in ps_list]
        logger.debug("Launched timeline gets")
        timelines = ray.get(timelines)
        t0 = timelines[0]
        for t in timelines[1:]:
            t0.merge(t)
        t0.chrome_trace_format("ps_timeline.json")
    else:
        # Wait for at least the ps gets to finish
        ray.get(ps_gets)
    if fetch_stats:
        return {"loss": np.mean(ray.get(losses))}
    else:
        return None
