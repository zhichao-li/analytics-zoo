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

package com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader.model

import com.fasterxml.jackson.databind.JsonNode
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.utils.Node
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.models.{Model, Sequential}
import com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader.{LayerLoader, Loader, ModelLoader, Utils}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object ModelLoader extends Loader {

  override def fromConfig(outterLevelConfig: JsonNode): AbstractModule[Activity, Activity, Float] = {
    var iter = outterLevelConfig.get("config").get("layers").iterator()
    val layers = ArrayBuffer[KerasLayer[Activity, Activity, Float]]()
    val nameToLayer = mutable.Map[String, KerasLayer[Activity, Activity, Float]]()
    val nameToNode = mutable.Map[String, ModuleNode[Float]]()
    while (iter.hasNext()) {
      val layer = LayerLoader.load(iter.next())
      if (!nameToLayer.contains(layer.getName())) {
        nameToLayer += (layer.getName() -> layer)
      } else {
        throw new RuntimeException(s"Found duplicated name: ${layer.getName()}")
      }
      layers.append(layer)
    }
    // Build the graph
    iter = outterLevelConfig.get("config").get("layers").iterator()
    while(iter.hasNext) {
      val config = iter.next()
      val layerName = config.get("name").asText()
      val inBoundNodes = Utils.toInboundNodes(config)
      require(inBoundNodes.length <= 1,
        s"we don't support share weights for now, " +
          s"but the input of ${layerName} is ${inBoundNodes.map(_._1)}")
      val inputNodes = inBoundNodes.map{in => nameToNode(in._1)}
      val layer = nameToLayer(layerName)
      val node: ModuleNode[Float] = if (layer.isInstanceOf[Input[Float]]) {
        layer.build(KerasUtils.addBatch(layer.asInstanceOf[Input[Float]].inputShape))
        new Node(layer.asInstanceOf[AbstractModule[Activity, Activity, Float]])
      } else {
        nameToLayer(layerName).inputs(inputNodes)
      }
      if (!nameToNode.contains(layerName)) {
        nameToNode += (layerName -> node)
      }
    }

    // Get the input nodes
    iter = outterLevelConfig.get("config").get("input_layers").iterator()
    val inputLayerBuf = ArrayBuffer[String]()
    while (iter.hasNext) {
      inputLayerBuf.append(iter.next().get(0).asText())
    }

    // Get the output nodes
    iter = outterLevelConfig.get("config").get("output_layers").iterator()
    val outLayerBuf = ArrayBuffer[String]()
    while (iter.hasNext) {
      outLayerBuf.append(iter.next().get(0).asText())
    }

    val model = Model[Float](input = inputLayerBuf.toArray.map(nameToNode(_)),
      output = outLayerBuf.toArray.map(nameToNode(_)))
    val name = outterLevelConfig.get("config").get("name").asText()
    model.setName(name)
    model
  }
}