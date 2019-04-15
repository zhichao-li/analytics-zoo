from ray_poc.allreduce.sgd import ModelWorker
import numpy as np

vector=np.ones([10])
result = ModelWorker.split(vector, 4)
assert len(result) == 4
assert len(result[0]) == 3
assert len(result[1]) == 3
assert len(result[2]) == 2
assert len(result[3]) == 2
