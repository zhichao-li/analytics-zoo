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
import com.intel.analytics.zoo.pipeline.api.keras.layers.InputLayer
import com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader.{LayerLoader, Utils}

object InputLayerLoader extends LayerLoader {

  override def toZooFormat(kerasWeights: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    throw new RuntimeException("Invalid operation")
  }

  def doFromConfig(layerLevelConfig: JsonNode): AbstractModule[Activity, Activity, Float] = {
    val config = layerLevelConfig.get("config")
    val inputShape = Utils.getInputShape(config)
    val module = InputLayer[Float](
      inputShape = inputShape).asInstanceOf[AbstractModule[Activity, Activity, Float]]
    module
  }
}
