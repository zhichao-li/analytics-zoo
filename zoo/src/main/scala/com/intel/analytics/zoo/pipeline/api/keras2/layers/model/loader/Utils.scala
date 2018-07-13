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

import com.fasterxml.jackson.core.`type`.TypeReference
import com.fasterxml.jackson.databind.{DeserializationFeature, JsonNode, ObjectMapper}
import com.intel.analytics.bigdl.utils.Shape

import scala.collection.mutable.ArrayBuffer

object Utils {

   def toJson(str: String): JsonNode = {
     val mapper = new ObjectMapper()
     mapper.enable(DeserializationFeature.FAIL_ON_READING_DUP_TREE_KEY)
     mapper.readTree(str)
   }

   def toArrayInt(json: JsonNode): Array[Int] = {
     val result = ArrayBuffer[Int]()
     val iter = json.iterator()
     while(iter.hasNext) {
       result.append(iter.next().asInt())
     }
     result.toArray
   }

  def toInboundNodes(jsonConfig: JsonNode): Array[(String, Int, Int)] = {
    if (jsonConfig.get("inbound_nodes") == null) {
      return Array[(String, Int, Int)]()
    }
    val iter = jsonConfig.get("inbound_nodes").iterator()
    val result = ArrayBuffer[(String, Int, Int)]()
    while(iter.hasNext) {
      val node = iter.next()
      val tensors = node.iterator()
      var i = 0
      while(tensors.hasNext) {
        i += 1
        val tensor = tensors.next()
        val layerName = tensor.get(0).asText()
        val outIndex = tensor.get(1).asInt()
        val outTensorIndex = tensor.get(2).asInt()
        result.append((layerName, outIndex, outTensorIndex))
      }
      // TODO: still no idea why there's an array here.
      require(i == 1, s"the items within node should be 1, but got: ${i}")
    }
    result.toArray
  }

  def getInputShape(json: JsonNode): Shape = {
    if (json == null || json.get("batch_input_shape") == null) {
      null
    } else {
      val batchShape = toArrayInt(json.get("batch_input_shape"))
      require(batchShape.length >= 2,
        s"batch_input_shape should >=2, but got: ${batchShape.length}")
      Shape(batchShape.drop(1))
    }
  }

//  def jsonToMap(str: String): JsonNode = {
//    val mapper = new ObjectMapper()
//    mapper.enable(DeserializationFeature.FAIL_ON_READING_DUP_TREE_KEY)
//    mapper.readValue(str, new TypeReference[Map[String, String]](){})
//  }

}
