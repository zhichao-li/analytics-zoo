import tensorflow as tf
import numpy as np

from ray_poc.allreduce.tfutils import GVHelper
from ray_poc.tfutils import TensorFlowVariables


class RayModel(object):
    def __init__(self, input_ops, label_ops, num_worker, optimizer):
        self.num_worker = num_worker
        self.optimizer = optimizer
        self.input_ops = input_ops
        self.label_ops = label_ops
        # TODO: expose config here.
        # self.sess = tf.Session(
        #     config=tf.ConfigProto(
        #         intra_op_parallelism_threads=1,
        #         inter_op_parallelism_threads=1))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.loss_op = self._get_loss_op()
        self.optimizer = self._get_optimizer()


        # A list of (gradient, variable) pairs
        self.grad_vars = [
            t for t in optimizer.compute_gradients(self.loss_op)
            if t[0] is not None
        ]
        self.gv_helper = GVHelper(
            sess=self.sess, grad_vars=self.grad_vars)


        # self.gholders_vars = [(tf.placeholder(
        #     "float", shape=grad[1].get_shape()), grad[1])
        #                           for grad in self.grads_vars]
        # self.apply_grads_placeholder = self.optimizer.apply_gradients(
        #     self.gholders_vars)

    # is this run in multi-thread way? otherwise we need to put it in PS, but we need to implement the update logic case by case
    # or each ps would be responsible for a small collection of (grads, vars) instead of a single vector?
    # if we use flat_gradient, grads -flat- -> gvector -agg---split--> apply
    # NB: not used for now as we do this in PS
    # def apply_gradients(self, gradients):
    #     """
    #     This gradient should have been correctly splitted according to the vars.
    #     """
    #     feed_dict = {}
    #     for i in range(len(self.gholders_vars)):
    #         feed_dict[self.gholders_vars[i][0]] = gradients[i]
    #     self.sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

    def compute_gradients(self, inputs, labels):
        """
        :param inputs:
        :param labels:
        :return: The returning gradient is not a flat gradient and it's divided by vars
        """
        assert isinstance(inputs, list)
        assert isinstance(labels, list)
        fdict = {}
        fdict.update(dict(zip(self.input_ops, inputs)))
        fdict.update(dict(zip(self.label_ops, labels)))
        # TODO: what if the feed data is more than inputs and labels? i.e
        # return self.sess.run(
        #     [grad[0] for grad in self.grads],
        #     feed_dict={
        #         self.x: x,
        #         self.y_: y,
        #         self.keep_prob: 0.5
        #     })
        gradients = self.sess.run(
            [grad[0] for grad in self.grad_vars],
            feed_dict=fdict)
        return gradients

    def _get_loss_op(self):
        raise Exception("not supported yet")

    def _get_optimizer(self):
        raise Exception("not supported yet")

    def set_flat_parameters(self, parameters):
        """
        :param parameters: 1D vector
        :return:
        """
        assert len(parameters.shape) == 1, "we only accept 1D vector here, but got: {}".format(len(parameters.shape))
        self.gv_helper.set_flat(parameters)

    # tfvariables.variables is a OrderDict, so you don't need to worried about the order.
    def get_flat_parameters(self):
        return self.gv_helper.get_flat()


class ClassicTFRayModel(RayModel):
    def __init__(self, loss_op, optimizer, input_ops, label_ops, num_worker):
        self.loss_op = loss_op

        def to_list(a):
            if not isinstance(a, list):
                return [a]
            else:
                return a
        input_ops = to_list(input_ops)
        label_ops = to_list(label_ops)
        super(ClassicTFRayModel, self).__init__(input_ops=input_ops, label_ops=label_ops, num_worker=num_worker, optimizer=optimizer)


    def _get_loss_op(self):
        return self.loss_op

    def _get_optimizer(self):
        return self.optimizer



class KerasTFRayModel(RayModel):
    def __init__(self, keras_model, keras_optimizer):
        raise Exception("not supported yet")



