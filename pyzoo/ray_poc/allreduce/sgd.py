from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import random
import time

import numpy as np

import ray
from ray.experimental.sgd.sgd_worker import SGDWorker
from ray.experimental.sgd.param_server import ParameterServer

logger = logging.getLogger(__name__)


class DistributedSGD(object):

    def __init__(self,
                 model,
                 num_workers):

        requests = {"num_cpus": 1}

        RemoteSGDWorker = ray.remote(**requests)(SGDWorker)
        self.workers = []
        logger.info(
            "Creating SGD workers ({} total)".format(
                num_workers))
        for worker_index in range(num_workers):
            self.workers.append(
                RemoteSGDWorker.remote(
                    worker_index,
                    model_creator))

        logger.info("Waiting for gradient configuration")
        shard_shapes = ray.get(self.workers[0].shard_shapes.remote())

        logger.info("Waiting for actors to start")
        ray.get([w.shard_shapes.remote() for w in self.workers])

        if strategy == "ps":
            logger.info("Starting parameter servers ({} shards)".format(
                len(shard_shapes)))
            self.ps_list = [
                ParameterServer.remote(len(self.workers), i)
                for i, s in enumerate(shard_shapes)
            ]
            ray.get([
                ps.initialize.remote(s)
                for ps, s in zip(self.ps_list, shard_shapes)
            ])
            logger.info("Parameter servers started")
        else:
            self.ps_list = []

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
