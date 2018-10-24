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

package com.intel.analytics.zoo.aep

import org.apache.spark.unsafe.Platform

import scala.collection.mutable.ArrayBuffer

case class Bytes(value: Array[Byte])


class AEPBytesArray(val startAddr: Long, val size: Long, val sizeOfBytes: Int) extends AEPArray[Bytes](startAddr, size) {

//  override def get(i: Long): Bytes = {
//    assert(!deleted)
//    val result = ArrayBuffer[Byte]()
//    val startOffset = indexOf(i)
//    var j = 0
//    while(j < sizeOfBytes) {
//      result.append(Platform.getByte(null, startOffset + j))
//      j += 1
//    }
//    return Bytes(result.toArray)
//  }

  override def get(i: Long): Bytes = {
    val result = new Array[Byte](sizeOfBytes)
    Platform.copyMemory(null, indexOf(i), result, Platform.BYTE_ARRAY_OFFSET, sizeOfBytes)
    return Bytes(result)
  }

  def getMoveSteps(): Int = sizeOfBytes

  // TODO: would be slow if we put byte one by one.
  def set(i: Long, bytes: Bytes): Unit = {
    assert(!deleted)
    val startOffset = indexOf(i)
    var j = 0
    while(j < bytes.value.length) {
      Platform.putByte(null, startOffset + j, bytes.value(j))
      j += 1
    }
  }
}

