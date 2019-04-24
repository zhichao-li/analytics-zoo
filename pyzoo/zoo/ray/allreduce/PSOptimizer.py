import tensorflow as tf

from zoo.ray.allreduce.tfutils import GVHelper


class PSOptimizer(object):
    def __init__(self, num_worker, optimizer, loss_op):
        self.num_worker = num_worker
        self.optimizer = optimizer
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.loss_op = loss_op
        self.optimizer = optimizer

        # A list of (gradient, variable) pairs
        self.grad_vars = [
            t for t in optimizer.compute_gradients(self.loss_op)
            if t[0] is not None
        ]
        self.gv_helper = GVHelper(
            sess=self.sess, grad_vars=self.grad_vars)

    def compute_gradients(self, feed_data):
        gradients_loss = self.sess.run(
            [grad[0] for grad in self.grad_vars],
            feed_dict=feed_data)
        return gradients_loss

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

    def minimize(self):
        pass


# class ClassicTFRayModel(RayModel):
#     def __init__(self, loss_op, optimizer, input_ops, label_ops, y_pred_ops, num_worker):
#         self.loss_op = loss_op
#
#         def to_list(a):
#             if not isinstance(a, list):
#                 return [a]
#             else:
#                 return a
#         input_ops = to_list(input_ops)
#         label_ops = to_list(label_ops)
#         y_pred_ops = to_list(y_pred_ops)
#         super(ClassicTFRayModel, self).__init__(input_ops=input_ops, label_ops=label_ops, y_pred_ops=y_pred_ops, num_worker=num_worker, optimizer=optimizer)
#
#
#     def _get_loss_op(self):
#         return self.loss_op
#
#     def _get_optimizer(self):
#         return self.optimizer

