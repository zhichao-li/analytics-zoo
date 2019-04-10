import tensorflow as tf
import numpy as np

from ray_poc.tfutils import TensorFlowVariables


class RayModel(object):
    def __init__(self):
        # TODO: expose config here.
        self.sess = tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1))
        self.sess.run(tf.global_variables_initializer())
        self.loss_op = self._get_loss_output()
        self.optimizer = self._get_optimizer()

        self.variables = TensorFlowVariables(
            self.loss_op, self.sess)
        # A list of (gradient, variable) pairs
        self.grads_vars = self.optimizer.compute_gradients(self.loss_op)
        self.gholders_vars = [(tf.placeholder(
            "float", shape=grad[1].get_shape()), grad[1])
                                  for grad in self.grads]
        self.apply_grads_placeholder = self.optimizer.apply_gradients(
            self.gholders_vars)

    def _get_loss_output(self):
        raise Exception("not supported yet")

    def _get_optimizer(self):
        raise Exception("not supported yet")

    def compute(self, input):
        """
        forward and backward and return sharded_grads
        :return:
        """
        pass

    def set_flat_parameters(self, parameters):
        """
        :param parameters: 1D vector
        :return:
        """
        self.variables.set_flat()

    # tfvariables.variables is a OrderDict, so you don't need to worried about the order.
    def get_flat_parameters(self):
        self.variables.get_flat()

    def set_parameters(self, variable_names, weights):
        self.variables.set_weights(dict(zip(variable_names, weights)))

    def get_parameters(self):
        weights = self.variables.get_weights()
        return list(weights.keys()), list(weights.values())
