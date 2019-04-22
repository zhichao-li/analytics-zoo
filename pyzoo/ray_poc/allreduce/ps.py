import tensorflow as tf
import numpy as np
import ray

@ray.remote(resources={"ps":1})
class ShardedParameterServer(object):
    # ray_model would automatically serde here?
    def __init__(self, parameters, gen_ray_model):
        """
        :param parameters: 1D ndarray
        :param optimizer:
        """
        # TODO: we only need to pass a gen_optimizer here.
        ray_model = gen_ray_model()
        self.optimizer=ray_model.optimizer

        self.parameters=parameters
        self.grad_holder = tf.placeholder(
            tf.float32,
            self.parameters.shape,
            name="Placeholder_grads")

        self.weight_var = tf.Variable(
            initial_value=parameters,
            dtype=tf.float32,
            name="variable_weights")

        self.apply_op = self.optimizer.apply_gradients([(self.grad_holder, self.weight_var)])
        # TODO: turn the parameter here?
        # sess = tf.Session(
        #     config=tf.ConfigProto(
        #         intra_op_parallelism_threads=1,
        #         inter_op_parallelism_threads=1))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.parameters = parameters

    def get_parameters(self):
        # raise Exception("bad bad")
        return self.parameters


    def apply_gradients(self, *gradients):
        """
        :param gradients:
        :return: updated weights
        """
        # TODO: MKL here?
        agg_grad = np.mean(gradients, axis=0)

        _, parameters = self.sess.run([self.apply_op, self.weight_var], feed_dict={self.grad_holder: agg_grad})
        self.parameters = parameters
        # TODO: maybe we can return a metrics here.
        return "success"


class ShardedParameterServerForGradWeights(object):
    pass


    # def aggregate_grads(self, *grads):
    #     """
    #     :param grads: list of sharded gradients
    #     :return:
    #     """
    #     # calculate the mean of grads
    #     grads = np.average(grads, axis=0)
    #     # update weights
    #     # TODO: we should have an optimizer here for that
    #     self.parameters += grads