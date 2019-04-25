
class RayDataSet(object):
    pass

class DummyRayDataSet(RayDataSet):
    def __init__(self, feature_shape, label_shape):
        self.feature_shape=feature_shape
        self.label_shape=label_shape

    # it should return list of inputs and list of labels
    def next_batch(self):
        return [np.random.uniform(0, 1, size=self.feature_shape)], [np.random.uniform(0, 1, size=self.label_shape)]