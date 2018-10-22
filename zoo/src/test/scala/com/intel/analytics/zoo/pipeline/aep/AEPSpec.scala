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

package com.intel.analytics.zoo.pipeline.aep

import com.intel.analytics.zoo.aep.{AEPBytesArray, AEPFloatArray, AEPHandler, Bytes}
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper

import scala.collection.mutable.ArrayBuffer

class AEPSpec extends ZooSpecHelper {
  "load native lib" should "be ok" in {
    val address = AEPHandler.allocate(1000L)
    AEPHandler.free(address)
  }

  "AEPFloatArray" should "be ok" in {
    val address = AEPHandler.allocate(1000)
    AEPHandler.free(address)
    val array = Array[Float](1.2f, 0.3f, 4.5f, 199999.6f)
    val aepArray = AEPFloatArray(array.toIterator, array.size)
    var i = 0
    while( i < aepArray.size) {
      assert(aepArray.get(i) == array(i))
      i += 1
    }
  }

  "AEPBytesArray" should "be ok" in {
    val sizeOfItem = 100
    val sizeOfRecord = 5
    val addr = AEPHandler.allocate(sizeOfItem * sizeOfRecord)
    val aepArray = new AEPBytesArray(addr, sizeOfItem, sizeOfRecord)
    val targetArray = ArrayBuffer[Byte]()
    val rec = Bytes(Array[Byte](193.toByte, 169.toByte, 0, 90, 4))
    (0 to 100).foreach {i =>
      aepArray.set(i, rec)
    }

    var i = 0
    while( i < sizeOfItem) {
      assert(aepArray.get(i).value === rec.value)
      i += 1
    }
  }
}
