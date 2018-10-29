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
import com.intel.analytics.zoo.aep.{AEPBytesArray, AEPFloatArray, AEPHandler}
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import org.apache.hadoop.io.Text
import org.apache.spark.SparkContext

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
    aepArray.free()
  }

  "AEPBytesArray" should "be ok" in {
    val sizeOfItem = 100
    val sizeOfRecord = 5
    val aepArray = AEPBytesArray(sizeOfItem, sizeOfRecord)
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
    // TODO: there's problem when invoke free here.
    aepArray.free()
  }

  "cached imageset" should "be ok" in {
    def readLabel(data: Text): String = {
      val dataArr = data.toString.split("\n")
      if (dataArr.length == 1) {
        dataArr(0)
      } else {
        dataArr(1)
      }
    }
    val classNum = 1000
    val conf = Engine.createSparkConf().setAppName("BigDL InceptionV1 Train Example")
      .set("spark.task.maxFailures", "1").setMaster("local[2]")
    val sc = new SparkContext(conf)
    Engine.init

    val nodeNumber = 1
    val coreNumber = 4
    val resource = getClass.getClassLoader.getResource("aep/mini_imagenet_seq")
    val dataPath = "/home/lizhichao/data/imagenet-noresize-super-small" //resource.getPath
    val rawData = sc.sequenceFile(dataPath, classOf[Text],
      classOf[Text],
      nodeNumber * coreNumber)
      .map(image => {
      ByteRecord(image._2.copyBytes(), readLabel(image._1).toFloat)
    }).filter(_.label <= classNum)
//    val dataSet = DataSet.rdd(rawData)

    val dataSet = ImageSet.cacheWithAEP(rawData)
    val num = dataSet.data(train = false).count
    assert(num == 5)

    val imageSize = 224
    val batchSize = 1
    val transformedDataSet = dataSet.transform(
      MTLabeledBGRImgToBatch[ByteRecord](
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = (BytesToBGRImg() ) -> BGRImgCropper(imageSize, imageSize)
          //-> HFlip(0.5) -> BGRImgNormalizer(0.485, 0.456, //0.406, 0.229, 0.224, 0.225)
      ))
    val ddd =transformedDataSet.asInstanceOf[DistributedDataSet[MiniBatch[Float]]].data(train =
      false).count()
    ddd
  }
}
