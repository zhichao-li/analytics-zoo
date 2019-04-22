from zoo.ray.allreduce.sgd import DistributedOptimizer
from zoo.ray.test.allreduce.test_resnet import dataset_fn, classic_tf_fn


def test_resnet_ps(num_worker):
    batch_size = 64 # NB if you want to change this you need to change classic_tf_fn as well
    dopt = DistributedOptimizer.from_classic_tf(model_fn=classic_tf_fn, dataset_fn=dataset_fn, batch_size=batch_size, num_worker=num_worker)
