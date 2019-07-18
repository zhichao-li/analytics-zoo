#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tensorflow as tf
import numpy as np
import ray
import os
import time


from zoo.ray.util.utils import MKLSetting
from zoo.ray.util import utils


@ray.remote(resources={"trainer":1})
class ModelWorker(object):
    def __init__(self, modelLite, ray_data_set):
        self.modelAdapter = modelLite.to_adapter()
        self.ray_data_set = ray_data_set.action()
        self.loss = 0
        self.gradient = None
        self.training_grads = None
        # TODO: Move this to remote function
        self.num_ps = None
        self.num_models_per_node=None
    def get_num_ps(self):
        return self.num_ps

    def get_num_models_per_node(self):
        return self.num_models_per_node


    def set_num_ps(self, num_ps):
        self.num_ps = num_ps

    def set_num_models_per_node(self, num):
        self.num_models_per_node=num

    def pull(self):
        return self.gradient

    def push(self, *gradients):
        self.gradient = np.mean(gradients, axis=0)
        return 0

    # TODO: pass in num_splits here.
    def concate_and_split(self, *gradients):
        flat_grads = np.concatenate(gradients)
        return utils.split(flat_grads, self.num_ps)


    def ip(self):
        import ray.services as rservices
        return rservices.get_node_ip_address()

    # @ray.remote(num_return_vals=2)
    def pull_and_execute(self, *parameters):
        """
        It would return a sharded grads here.
        Each parameter should be a 1-D vector
        """
        start = time.time()
        flat_parameters = np.concatenate(parameters)
        self.modelAdapter.set_flat_trainable_weights(flat_parameters)
        set_weight_end = time.time()
        input_data, label_data = self.ray_data_set.next_batch()
        get_data_end = time.time()
        loss_gradients = self.modelAdapter.execute(utils.to_list(input_data),
                                                   utils.to_list(label_data))
        compute_end = time.time()
        self.loss = loss_gradients[0]
        self.training_grads = loss_gradients[1:]
        print("loss is {}".format(self.loss))
        flat_grads = np.concatenate([g.flatten() for g in self.training_grads])
        print("flat_grads {}".format(flat_grads.shape))
        sharded_grads = utils.split(flat_grads, self.num_models_per_node)
        end = time.time()
        print("set_weight: {}".format(set_weight_end - start))
        print("get_data_end: {}".format(get_data_end - set_weight_end))
        print("compute end: {}".format(compute_end - get_data_end))
        print("Time for pull and execute: {}".format(end - start))
        return sharded_grads


    # def pull_and_execute(self, ):

    def get_loss(self):
        # TODO: this should be combined with set_parameters_compute_gradients, but we cannot return
        # in this way (loss, List(grad1, grad2))
        return self.loss

    def get_flat_trainable_weights(self):
        return self.modelAdapter.get_flat_weights()

