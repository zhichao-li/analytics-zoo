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

object AEPBytesArray {
  def apply(iterator: Iterator[Array[Byte]], recordNumber: Int, recordBytes: Int): AEPBytesArray = {
    val aepArray = AEPBytesArray(recordNumber, recordBytes)
    var i = 0
    while(iterator.hasNext) {
      aepArray.set(i, iterator.next())
      i += 1
    }
    aepArray
  }

  def apply(recordNumber: Int, recordBytes: Int): AEPBytesArray = {
//        val startAddr = AEPHandler.allocate(recordNumber * recordBytes)
    val startAddr: Long = Platform.allocateMemory(recordNumber * recordBytes)
    assert(startAddr > 0, "Not enough memory!")
    new AEPBytesArray(startAddr, recordNumber, recordBytes)
  }


}
// length + content? var length of record.
class AEPBytesArray(val startAddr: Long, val size: Long, val sizeOfBytes: Int) extends AEPArray[Array[Byte]](startAddr, size) {

//  override def get(i: Long): Array[Byte] = {
//    assert(!deleted)
//    val result = ArrayBuffer[Byte]()
//    val startOffset = indexOf(i)
//    var j = 0
//    while(j < sizeOfBytes) {
//      result.append(Platform.getByte(null, startOffset + j))
//      j += 1
//    }
//    return result.toArray
//  }

  override def get(i: Long): Array[Byte] = {
    val result = new Array[Byte](sizeOfBytes)
    Platform.copyMemory(null, indexOf(i), result, Platform.BYTE_ARRAY_OFFSET, sizeOfBytes)
    return result
  }

  def getMoveSteps(): Int = sizeOfBytes

  // TODO: would be slow if we put byte one by one.
  def set(i: Long, bytes: Array[Byte]): Unit = {
    assert(!deleted)
    val startOffset = indexOf(i)
    var j = 0
    while(j < bytes.length) {
      Platform.putByte(null, startOffset + j, bytes(j))
      j += 1
    }
  }
}

