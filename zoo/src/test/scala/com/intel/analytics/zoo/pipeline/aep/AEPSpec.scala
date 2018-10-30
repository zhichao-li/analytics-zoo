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

import com.intel.analytics.bigdl.dataset.image.{HFlip, _}
import com.intel.analytics.bigdl.dataset.{ByteRecord, DataSet, DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.zoo.aep.{AEPBytesArray, AEPFloatArray, AEPHandler, AEPVarBytesArray}
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.image.inception.ImageNet2012
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext

import scala.collection.mutable.ArrayBuffer

class AEPSpec extends ZooSpecHelper {
  var sc: SparkContext = null

  override def doBefore(): Unit = {
    val conf = Engine.createSparkConf().setAppName("AEPSpec")
      .set("spark.task.maxFailures", "1").setMaster("local[4]")
    sc = NNContext.initNNContext(conf)
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

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
    while( i < aepArray.recordNum) {
      assert(aepArray.get(i) == array(i))
      i += 1
    }
    aepArray.free()
  }

  "AEPBytesArray" should "be ok" in {
    val sizeOfItem = 100
    val sizeOfRecord = 5
    val aepArray = new AEPBytesArray(sizeOfItem, sizeOfRecord)
    val targetArray = ArrayBuffer[Byte]()
    val rec = Array[Byte](193.toByte, 169.toByte, 0, 90, 4)
    (0 until 100).foreach {i =>
      aepArray.set(i, rec)
    }

    var i = 0
    while( i < sizeOfItem) {
      assert(aepArray.get(i) === rec)
      i += 1
      println(i)
    }
    aepArray.free()
  }

  "AEPvarBytesArray" should "be ok" in {
    val aepArray = new AEPVarBytesArray(3, 5 + 2 + 6)
    val targetArray = ArrayBuffer[Byte]()
    val rec1 = Array[Byte](193.toByte, 169.toByte, 0, 90, 4)
    val rec2 = Array[Byte](90, 4)
    val rec3 = Array[Byte](193.toByte, 169.toByte, 0, 90, 4, 5)

    aepArray.set(0, rec1)
    aepArray.set(1, rec2)
    aepArray.set(2, rec3)
    aepArray.free()
  }


  "cached imageset" should "be ok" in {

    val dataPath = getClass.getClassLoader.getResource("aep/mini_imagenet_seq").getPath

    val imageNet = ImageNet2012(path = dataPath,
      sc = sc,
      imageSize = 224,
      batchSize = 2,
      nodeNumber = 1,
      coresPerNode = 4,
      classNumber = 1000,
      cacheWithAEP = true).asInstanceOf[DistributedDataSet[MiniBatch[Float]]]
    val data = imageNet.data(train = false)
    assert(data.count() == 3)
    data.collect()
  }
}
