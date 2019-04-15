from ray_poc.allreduce.sgd import DistributedOptimizer, DummyRayDataSet
from ray_poc.test.allreduce.model import create_classic_tf_model
import ray
import tensorflow as tf

def dataset_fn():
    return DummyRayDataSet(feature_shape=[32, 784], label_shape=[32, 10])

ray.init(local_mode=True)
dopt = DistributedOptimizer.from_classic_tf(model_fn=create_classic_tf_model, dataset_fn=dataset_fn, num_worker=2)