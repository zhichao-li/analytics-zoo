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

package com.intel.analytics.zoo.pipeline.api.load.model

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Keras2Test, KerasBaseSpec}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import com.intel.analytics.zoo.pipeline.api.model.loader.HDF5Reader

import scala.util.Random

class ModelLoadingSpec extends KerasBaseSpec {

  "Read hdf5" should "be ok" taggedAs(Keras2Test) in {
    val hdf5Reader = new HDF5Reader("/home/lizhichao/bin/god/zoo/seq.h5")
    val modelConfig = hdf5Reader.readAttributeAsFixedLengthString("model_config", 100)
    println(modelConfig)

  }

}

