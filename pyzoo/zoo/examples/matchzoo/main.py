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
from __future__ import print_function
import os
import sys
import time
import json
import argparse
import random
random.seed(49999)
import numpy
numpy.random.seed(49999)
import tensorflow
tensorflow.set_random_seed(49999)

from collections import OrderedDict

from zoo.pipeline.api.autograd import *
from zoo.pipeline.api.keras.layers import *
from zoo.pipeline.api.keras.models import *
from bigdl.keras.converter import WeightsConverter
from test.zoo.pipeline.utils.test_utils import ZooTestCase
import numpy as np

from utils import *
import inputs
import metrics

np.random.seed(1330)

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.Session(config = config)

def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    model_config = config['model']['setting']
    model_config.update(config['inputs']['share'])
    sys.path.insert(0, config['model']['model_path'])

    model = import_object(config['model']['model_py'], model_config)
    mo = model.build()
    return mo

def load_keras2_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    model_config = config['keras2model']['setting']
    model_config.update(config['inputs']['share'])
    sys.path.insert(0, config['keras2model']['model_path'])

    model = import_object(config['keras2model']['model_py'], model_config)
    mo = model.build()
    return mo



def standardize_input_data(data, names, shapes=None,
                           check_batch_axis=True,
                           exception_prefix=''
                           ):
    if not names:
        if data is not None and hasattr(data, '__len__') and len(data):
            raise ValueError('Error when checking model ' +
                             exception_prefix + ': '
                             'expected no data, but got:', data)
        return []
    if data is None:
        return [None for _ in range(len(names))]

    if isinstance(data, dict):
        try:
            data = [data[x].values if data[x].__class__.__name__ == 'DataFrame' else data[x] for x in names]
        except KeyError as e:
            raise ValueError(
                'No data provided for "' + e.args[0] + '". Need data '
                'for each key in: ' + str(names))
    elif isinstance(data, list):
        if len(names) == 1 and data and isinstance(data[0], (float, int)):
            data = [np.asarray(data)]
        else:
            data = [x.values if x.__class__.__name__ == 'DataFrame' else x for x in data]
    else:
        data = data.values if data.__class__.__name__ == 'DataFrame' else data
        data = [data]
    data = [np.expand_dims(x, 1) if x is not None and x.ndim == 1 else x for x in data]

    if len(data) != len(names):
        if data and hasattr(data[0], 'shape'):
            raise ValueError(
                'Error when checking model ' + exception_prefix +
                ': the list of Numpy arrays that you are passing to '
                'your model is not the size the model expected. '
                'Expected to see ' + str(len(names)) + ' array(s), '
                'but instead got the following list of ' +
                str(len(data)) + ' arrays: ' + str(data)[:200] + '...')
        elif len(names) > 1:
            raise ValueError(
                'Error when checking model ' + exception_prefix +
                ': you are passing a list as input to your model, '
                'but the model expects a list of ' + str(len(names)) +
                ' Numpy arrays instead. The list you passed was: ' +
                str(data)[:200])
        elif len(data) == 1 and not hasattr(data[0], 'shape'):
            raise TypeError(
                'Error when checking model ' + exception_prefix +
                ': data should be a Numpy array, or list/dict of '
                'Numpy arrays. Found: ' + str(data)[:200] + '...')
        elif len(names) == 1:
            data = [np.asarray(data)]

    # Check shapes compatibility.
    if shapes:
        for i in range(len(names)):
            if shapes[i] is not None:
                data_shape = data[i].shape
                shape = shapes[i]
                if data[i].ndim != len(shape):
                    raise ValueError(
                        'Error when checking ' + exception_prefix +
                        ': expected ' + names[i] + ' to have ' +
                        str(len(shape)) + ' dimensions, but got array '
                        'with shape ' + str(data_shape))
                if not check_batch_axis:
                    data_shape = data_shape[1:]
                    shape = shape[1:]
                for dim, ref_dim in zip(data_shape, shape):
                    if ref_dim != dim and ref_dim:
                        raise ValueError(
                            'Error when checking ' + exception_prefix +
                            ': expected ' + names[i] + ' to have shape ' +
                            str(shape) + ' but got array with shape ' +
                            str(data_shape))
    return data


def predict(config):
    ######## Read input config ########


    model = load_model(config)
    # model.load_weights(weights_file)
    keras2_model = load_keras2_model(config)

    ######## Get and Set Weights ########
    query_embedding = keras2_model.get_layer("query_embedding")
    query_weights = query_embedding.get_weights()
    zquery_weights = WeightsConverter.to_bigdl_weights(query_embedding, query_weights)
    zquery_embedding = [l for l in model.layers if l.name() == "query_embedding"][0]
    zquery_embedding.set_weights(zquery_weights)

    # doc_embedding = keras2_model.get_layer("doc_embedding")
    # doc_weights = doc_embedding.get_weights()
    # zdoc_weights = WeightsConverter.to_bigdl_weights(doc_embedding, doc_weights)
    # zdoc_embedding = [l for l in model.layers if l.name() == "doc_embedding"][0]
    # zdoc_embedding.set_weights(zdoc_weights)

    dense = keras2_model.get_layer("dense")
    dense_weights = dense.get_weights()
    zdense_weights = WeightsConverter.to_bigdl_weights(dense, dense_weights)
    zdense = [l for l in model.layers if l.name() == "dense"][0]
    zdense.set_weights(zdense_weights)



    batch_size = 200
    query_data = np.random.randint(0, 10000, [batch_size, 10])
    doc_data = np.random.randint(0, 10000, [batch_size, 40])
    input_data = [query_data, doc_data]
    keras2_y_pred = keras2_model.predict(input_data, batch_size=batch_size)
    y_pred = model.forward(input_data)
    # y_pred = model.predict(input_data, distributed=False)

    equal = np.allclose(y_pred, keras2_y_pred, rtol=1e-5, atol=1e-5)
    equal
   # 16


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--model_file', default='./models/knrm_wikiqa.config', help='Model_file: MatchZoo model file for the chosen model.')
    args = parser.parse_args()
    model_file =  args.model_file
    with open(model_file, 'r') as f:
        config = json.load(f)
    phase = args.phase
    # if args.phase == 'train':
    #     train(config)
    # elif args.phase == 'predict':
    #     predict(config)
    # else:
    #     print('Phase Error.', end='\n')
    predict(config)
    return

if __name__=='__main__':
    main(sys.argv)
