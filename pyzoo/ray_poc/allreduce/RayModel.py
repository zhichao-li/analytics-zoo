import tensorflow as tf
import numpy as np

from ray_poc.allreduce.tfutils import GVHelper
from ray_poc.tfutils import TensorFlowVariables


class RayModel(object):
    def __init__(self, input_ops, label_ops, y_pred_ops, num_worker, optimizer):
        self.num_worker = num_worker
        self.optimizer = optimizer
        self.input_ops = input_ops
        self.label_ops = label_ops
        self.y_pred_ops = y_pred_ops
        # TODO: expose config here.
        # self.sess = tf.Session(
        #     config=tf.ConfigProto(
        #         intra_op_parallelism_threads=1,
        #         inter_op_parallelism_threads=1))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.loss_op = self._get_loss_op()
        self.optimizer = self._get_optimizer()

        self._gen_accuracy()

        # A list of (gradient, variable) pairs
        self.grad_vars = [
            t for t in optimizer.compute_gradients(self.loss_op)
            if t[0] is not None
        ]
        self.gv_helper = GVHelper(
            sess=self.sess, grad_vars=self.grad_vars)

    # we need to know the output of model not only loss
    # this is not a generic method, we need to think about how to deal with it.
    def _gen_accuracy(self):
        assert len(self.label_ops) == 1
        with tf.name_scope('accuracy'):
            # TODO: I suppose we only have one output and one label
            # label [-1, 1] not one-hot encoding. for tf.equal if the shape mismatch, then the result would be incorrect as it would broadcast automatically during the comparing stage.
            correct_prediction = tf.equal(tf.argmax(self.y_pred_ops[0], 1), tf.cast(tf.reshape(self.label_ops[0], (-1,)), tf.int64))
            self.correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(self.correct_prediction)


    # def compute_accuracy(self, inputs, labels):
    #         fdict = self._generate_feed_dict(inputs, labels)
    #         return self.sess.run(
    #             self.accuracy,
    #             feed_dict=fdict)


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

    def _generate_feed_dict(self, inputs, labels):
        assert isinstance(inputs, list)
        assert isinstance(labels, list)
        fdict = {}
        fdict.update(dict(zip(self.input_ops, inputs)))
        fdict.update(dict(zip(self.label_ops, labels)))
        return fdict

    def compute_gradients(self, inputs, labels):
        """
        :param inputs:
        :param labels:
        :return: The returning gradient is not a flat gradient and it's divided by vars
        """
        fdict = self._generate_feed_dict(inputs, labels)
        # TODO: what if the feed data is more than inputs and labels? i.e
        # return self.sess.run(
        #     [grad[0] for grad in self.grads],
        #     feed_dict={
        #         self.x: x,
        #         self.y_: y,
        #         self.keep_prob: 0.5
        #     })

        # gradients_loss = self.sess.run(
        #     [grad[0] for grad in self.grad_vars] + [self.accuracy, self.loss_op],
        #     feed_dict=fdict)
        # return gradients_loss[:-2], gradients_loss[-2], gradients_loss[-1]

        # TODO: make metric general
        gradients_loss = self.sess.run(
            [grad[0] for grad in self.grad_vars],
            feed_dict=fdict)
        return gradients_loss

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
    # The order is the same with optimizer.compute_gradient
    def get_flat_parameters(self):
        return self.gv_helper.get_flat()


class ClassicTFRayModel(RayModel):
    def __init__(self, loss_op, optimizer, input_ops, label_ops, y_pred_ops, num_worker):
        self.loss_op = loss_op

        def to_list(a):
            if not isinstance(a, list):
                return [a]
            else:
                return a
        input_ops = to_list(input_ops)
        label_ops = to_list(label_ops)
        y_pred_ops = to_list(y_pred_ops)
        super(ClassicTFRayModel, self).__init__(input_ops=input_ops, label_ops=label_ops, y_pred_ops=y_pred_ops, num_worker=num_worker, optimizer=optimizer)


    def _get_loss_op(self):
        return self.loss_op

    def _get_optimizer(self):
        return self.optimizer



class KerasTFRayModel(RayModel):
    def __init__(self, keras_model, keras_optimizer):
        raise Exception("not supported yet")



