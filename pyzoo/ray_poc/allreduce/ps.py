import numpy as np

class ParameterServerShard(object):
    def __init__(self, sharded_dim):
        self.parameters = np.zeros(sharded_dim)

    def get_parameters(self):
        return self.parameters

    def update_parameters(self, update):
        self.parameters += update


class ZooDistributedSGD():
    pass
