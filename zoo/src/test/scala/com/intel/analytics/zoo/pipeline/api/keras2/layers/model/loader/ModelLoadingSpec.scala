/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader

import com.intel.analytics.zoo.pipeline.api.keras.layers.{Keras2Test, KerasBaseSpec}
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Keras2ModelLoadingBaseSpec
import com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader.layers.DenseLoader

class ModelLoadingSpec extends Keras2ModelLoadingBaseSpec {

  "Read hdf5" should "be ok" taggedAs(Keras2Test) in {
    val hdf5Reader = new HDF5Reader("/home/lizhichao/bin/god/zoo/seq.h5")
    val modelConfig = hdf5Reader.readAttribute("model_config")
    val dataVaule = hdf5Reader.readDataSet("/model_weights/dense_1/dense_1/kernel:0")
    val str = Utils.toJson(modelConfig).get("config").get(0).get("config").toString
    println(modelConfig)
    val dense = DenseLoader.fromConfig(Utils.toJson(modelConfig).get("config").get(0).get("config"))
    dense
  }

  "LayerLoader" should "be ok" taggedAs(Keras2Test) in {
    val hdf5Reader = new HDF5Reader("/home/lizhichao/bin/god/zoo/seq.h5")
    val modelConfig = hdf5Reader.readAttribute("model_config")
    val layerConfig = Utils.toJson(modelConfig).get("config").get(0)
    LayerLoader.load(layerConfig)
  }

}

