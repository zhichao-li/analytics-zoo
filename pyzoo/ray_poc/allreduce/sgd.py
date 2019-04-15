from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import random
import time
import tensorflow as tf

import numpy as np

import ray

from ray_poc import split
from ray_poc.allreduce.ps import ShardedParameterServer

from ray_poc.allreduce.RayModel import ClassicTFRayModel, RayModel

logger = logging.getLogger(__name__)

class RayDataSet(object):
    pass

class DummyRayDataSet(RayDataSet):
    def __init__(self, feature_shape, label_shape):
        self.feature_shape=feature_shape
        self.label_shape=label_shape

    # it should return list of inputs and list of labels
    def next_batch(self):
        return [np.random.uniform(0, 1, size=self.feature_shape)], [np.random.uniform(0, 1, size=self.label_shape)],

@ray.remote(num_cpus=1)
class ModelWorker(object):

    def __init__(self, gen_ray_model, gen_ray_data_set, num_workers):
        # TODO: add a factory method for ModelWorker

        self.num_workers = num_workers
        self.ray_model = gen_ray_model()
        self.ray_data_set = gen_ray_data_set()

    def compute_gradients(self, parameters):
        """
        It would return a sharded grads here.
        Each parameter should be a 1-D vector
        """
        # concate grads
        flat_parameters = np.concatenate(parameters)
        self.ray_model.set_flat_parameters(flat_parameters)

        input, label = self.ray_data_set.next_batch()
        grads = self.ray_model.compute_gradients(input, label)
        flat_grads = np.concatenate([g.flatten() for g in grads])
        sharded_grads = split(flat_grads, self.num_workers)
        return sharded_grads





class DistributedOptimizer(object):

    def __init__(self,
                 gen_ray_model,
                 gen_ray_data_set,
                 num_worker):
        self.num_worker = num_worker
        # requests = {"num_cpus": 1}

        self.ray_model = gen_ray_model()
        # ModelWorkerWithResource = ray.remote(**requests)(ModelWorker)

        weights = self.ray_model.get_flat_parameters()
        sharded_weights = np.split(weights, self.num_worker)
        # sync the parameters in PS
        sharded_weight_ids = [ray.put(w) for w in sharded_weights]

        self.workers = []
        self.pss = []
        logger.info(
            "Creating parameter server ({} total)".format(
                num_worker))
        def gen_opt():
            return tf.train.AdamOptimizer(0.1)
        for ps_index in range(num_worker):
            self.pss.append(
                ShardedParameterServer.remote(sharded_weight_ids[ps_index], gen_ray_model=gen_ray_model))

        logger.info(
            "Creating model workers ({} total)".format(
                num_worker))
        for worker_index in range(num_worker):
            self.workers.append(
                ModelWorker.remote(gen_ray_model, gen_ray_data_set, num_worker))

        steps = 10
        for step in range(0, steps):
            self.run_step()

    @classmethod
    def from_classic_tf(cls, model_fn, dataset_fn, num_worker):

        def ray_model_fn():
            loss, optimizer, input, label = model_fn()
            ray_model = ClassicTFRayModel(loss_op=loss,
                                          optimizer=optimizer,
                                          input_ops=input,
                                          label_ops=label,
                                          num_worker=num_worker)
            return ray_model
        return cls(gen_ray_model=ray_model_fn,
                 gen_ray_data_set=dataset_fn,
                    num_worker=num_worker)

    def run_step(self):
        # workers of sharded_grads
        sharded_grad_ids = []
        # pull weights from ps
        for worker in self.workers:
            # 1) pull the latest weights from ps
            parameters = [ps.get_parameters.remote() for ps in self.pss]
            # 2) compute the grads
            sharded_grad_ids.append(worker.compute_gradients.remote(parameters))

        # TODO: performance?
        grads_per_ps = list(zip(*sharded_grad_ids))
        assert len(grads_per_ps[0]) == self.num_worker, "we should get correct grads for each ps"
        # 3) push and aggregate grads on ps
        for index, grads in enumerate(grads_per_ps):
            self.pss[index].apply_gradients.remote(grads)


