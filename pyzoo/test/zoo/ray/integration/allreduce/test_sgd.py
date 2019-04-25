import ray

from zoo.ray import DistributedOptimizer, DummyRayDataSet
from zoo.ray import create_classic_tf_model


def dataset_fn():
    return DummyRayDataSet(feature_shape=[32, 784], label_shape=[32, 10])

ray.init(local_mode=True)
dopt = DistributedOptimizer.from_classic_tf(model_fn=create_classic_tf_model, dataset_fn=dataset_fn, num_worker=2)




