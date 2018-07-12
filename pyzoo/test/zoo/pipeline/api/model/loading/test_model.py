#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import shutil

from zoo.pipeline.api.keras2.layers import *
from zoo.pipeline.api.keras.models import *
from test.zoo.pipeline.utils.test_utils import ZooTestCase
import numpy as np

np.random.seed(1337)  # for reproducibility


class TestModelLoading(ZooTestCase):

    def test_create_model(self):
        from keras.models import Sequential
        from keras.layers import Dense
        model = Sequential()
        model.add(Dense(10, kernel_initializer="glorot_uniform", input_shape=(8, )))
        ss = Sequential()
        ss.add(Dense(11, input_shape=(10, )))
        ss.add(Dense(14))
        model.add(ss)
        model.add(Dense(12))

        from keras.models import load_model
        model.save('/tmp/seq.h5')  # creates a HDF5 file 'my_model.h5'
        load_model('/tmp/seq.h5')

if __name__ == "__main__":
    pytest.main([__file__])
