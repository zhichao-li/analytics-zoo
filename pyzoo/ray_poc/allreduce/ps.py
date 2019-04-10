import numpy as np

class ShardedParameterServer(object):
    def __init__(self, sharded_parameters):
        self.parameters = sharded_parameters

    def get_parameters(self):
        return self.parameters

    # TODO: I suppose the grads dependencies would be resolved in parallel
    # otherwise it would be a bottle net here.
    def aggregate_grads_update_weights(self, *grads):
        """
        :param grads: list of sharded gradients
        :return:
        """
        # calculate the mean of grads
        grads = np.average(grads, axis=0)
        # update weights
        # TODO: we should have an optimizer here for that
        self.parameters += grads

