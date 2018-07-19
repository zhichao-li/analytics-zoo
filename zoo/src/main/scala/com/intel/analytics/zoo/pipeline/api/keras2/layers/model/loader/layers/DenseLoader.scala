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

package com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader.layers

import com.fasterxml.jackson.databind.JsonNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Dense
import com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader.{HDF5Reader, LayerLoader, Loader, Utils}

object DenseLoader extends LayerLoader {

  override def toZooFormat(kerasWeights: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    Array(kerasWeights(0).t(), kerasWeights(1))
  }

  override def doFromConfig(layerLevelConfig: JsonNode): AbstractModule[Activity, Activity, Float] = {
    val config = layerLevelConfig.get("config")
    val units = config.get("units").asInt()
    val activation = config.get("activation").asText()
    val useBias = config.get("use_bias").asBoolean()
    val kernelInitializer = config.get("kernel_initializer").asText()
    val biasInitializer = config.get("bias_initializer").asText()
    val kernelRegularizer = config.get("kernel_regularizer").asText()
    val biasRegularizer = config.get("kernel_regularizer").asText()
    val inputShape = Utils.getInputShape(config)

    val module = Dense[Float](units = units,
      kernelInitializer = kernelInitializer,
      biasInitializer = biasInitializer,
      activation = activation,
      kernelRegularizer = kernelRegularizer,
      biasRegularizer = biasRegularizer,
      useBias = useBias,
      inputShape = inputShape).asInstanceOf[AbstractModule[Activity, Activity, Float]]
    module
  }
}
