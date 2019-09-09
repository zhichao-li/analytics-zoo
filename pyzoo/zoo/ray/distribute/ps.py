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
import logging
import time

from zoo.ray.util.utils import MKLSetting
from zoo.ray.util import utils

logger = logging.getLogger(__name__)


@ray.remote(resources={"ps":1})
class ShardedParameterServer(MKLSetting):
    def __init__(self, parameters, modelLite, cores=None):
        """
        :param parameters: 1D ndarray for the initiate values
        :param optimizer:
        """
        super().__init__(cores=cores)
        modelAdapter = modelLite.to_adapter()
        self.optimizer=modelAdapter.optimizer

        self.parameters=parameters
        self.grad_holder = tf.placeholder(
            tf.float32,
            self.parameters.shape,
            name="ps_grads")

        self.weight_var = tf.Variable(
            initial_value=parameters,
            use_resource=True,
            dtype=tf.float32,
            name="ps_weights")

        self.apply_op = self.optimizer.apply_gradients([(self.grad_holder, self.weight_var)])

        self.sess = tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=self.get_cores(),
                inter_op_parallelism_threads=self.get_cores()))
        self.sess.run(tf.global_variables_initializer())
        self.parameters = parameters


    def pull(self):
        return self.parameters

    def push(self, *gradients):
        """
        :param gradients:
        :return: updated weights
        """
        #print(np.mean(gradients, axis=0).shape)
        mean_start = time.time()
        agg_grad = np.mean(gradients, axis=0).reshape(self.parameters.shape)
        mean_end = time.time()
        _, parameters = self.sess.run([self.apply_op, self.weight_var], feed_dict={self.grad_holder: agg_grad})
        self.parameters = parameters
        apply_grads_end = time.time()
        print(self.parameters.shape)
        print("Time for ps mean grads: {}".format(mean_end - mean_start))
        print("Time for ps apply_grads: {}".format(apply_grads_end - mean_end))
        return "success"
