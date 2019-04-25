import tensorflow as tf

from zoo.ray.allreduce.gvhelper import GVHelper
from zoo.ray.util import utils


class RayModel(object):
    """
    You should add your definition at model_fn
    and then return (input, output, target, loss, optimizer)
    """
    def __init__(self, model_fn):
        input, output, target, loss, optimizer = model_fn()
        self.optimizer = optimizer
        self.inputs = utils.to_list(input)
        self.outputs = utils.to_list(output)
        self.targets = utils.to_list(target)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.loss_op = loss
        self.optimizer = optimizer

        # A list of (gradient, variable) pairs
        self.grad_vars = [
            t for t in optimizer.compute_gradients(self.loss_op)
            if t[0] is not None
        ]
        self.gv_helper = GVHelper(
            sess=self.sess, grad_vars=self.grad_vars)

    @staticmethod
    def gen_model_fn(model_fn):
        def _ray_model():
            return RayModel(model_fn)
        return _ray_model

    def compute_gradients(self, feed_dict_data):
        """
        :param inputs:
        :return: The returning gradient is not a flat gradient and it's divided by vars
        """
        gradients = self.sess.run(
            [grad[0] for grad in self.grad_vars],
            feed_dict=feed_dict_data)
        return gradients


    def set_flat_parameters(self, parameters):
        """
        :param parameters: 1D vector
        :return:
        """
        assert len(parameters.shape) == 1, "we only accept 1D vector here, but got: {}".format(len(parameters.shape))
        self.gv_helper.set_flat(parameters)

    # The order is the same with optimizer.compute_gradient
    def get_flat_parameters(self):
        return self.gv_helper.get_flat()

