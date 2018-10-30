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

package com.intel.analytics.zoo.feature.image

import java.io.{FileOutputStream, ObjectOutputStream}
import java.nio.ByteBuffer
import java.util.concurrent.atomic.{AtomicInteger, AtomicLong}

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.{ByteRecord, CachedDistriDataSet, DataSet, DistributedDataSet}
import com.intel.analytics.bigdl.transform.vision.image.{DistributedImageFrame, ImageFeature, ImageFrame, LocalImageFrame}
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator}
import com.intel.analytics.zoo.aep.{AEPBytesArray, AEPVarBytesArray}
import com.intel.analytics.zoo.common.Utils
import com.intel.analytics.zoo.feature.common.Preprocessing
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.{EngineRef, KerasUtils}
import com.sun.jndi.cosnaming.IiopUrl.Address
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sparkExtension.SparkExtension
import org.apache.spark.storage.{BlockId, BlockManagerWrapper, StorageLevel}
import org.opencv.imgcodecs.Imgcodecs

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * ImageSet wraps a set of ImageFeature
 */
abstract class ImageSet {
  /**
   * transform ImageSet
   * @param transformer FeatureTransformer
   * @return transformed ImageSet
   */
  def transform(transformer: Preprocessing[ImageFeature, ImageFeature]): ImageSet

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> (transformer: Preprocessing[ImageFeature, ImageFeature]): ImageSet = {
    this.transform(transformer)
  }

  /**
   * whether this is a LocalImageSet
   */
  def isLocal(): Boolean

  /**
   * whether this is a DistributedImageSet
   */
  def isDistributed(): Boolean

  /**
   * return LocalImageSet
   */
  def toLocal(): LocalImageSet = this.asInstanceOf[LocalImageSet]

  /**
   * return DistributedImageSet
   */
  def toDistributed(): DistributedImageSet = this.asInstanceOf[DistributedImageSet]


  /**
   * Convert ImageFrame to ImageSet
   *
   * @return ImageSet
   */
  def toImageFrame(): ImageFrame

  /**
   * Convert ImageSet to DataSet of ImageFeature.
   */
  def toDataSet(): DataSet[ImageFeature]
}

class LocalImageSet(var array: Array[ImageFeature]) extends ImageSet {
  override def transform(transformer: Preprocessing[ImageFeature, ImageFeature]): ImageSet = {
    array = transformer.apply(array.toIterator).toArray
    this
  }

  override def isLocal(): Boolean = true

  override def isDistributed(): Boolean = false

  override def toImageFrame(): ImageFrame = {
    ImageFrame.array(array)
  }

  override def toDataSet(): DataSet[ImageFeature] = {
    DataSet.array(array)
  }
}

class DistributedImageSet(var rdd: RDD[ImageFeature]) extends ImageSet {
  override def transform(transformer: Preprocessing[ImageFeature, ImageFeature]): ImageSet = {
    rdd = transformer(rdd)
    this
  }

  override def isLocal(): Boolean = false

  override def isDistributed(): Boolean = true

  override def toImageFrame(): ImageFrame = {
    ImageFrame.rdd(rdd)
  }

  override def toDataSet(): DataSet[ImageFeature] = {
    DataSet.rdd[ImageFeature](rdd)
  }
}

abstract class ArrayLike[T: ClassTag] extends Serializable {
  def length: Int = throw new Error()

  def apply(i: Int): T = throw new Error()
}

case class AEPImageArray(imgs: AEPVarBytesArray, label: Array[Float]) extends ArrayLike[ByteRecord] {
  override def length: Int = {
    imgs.recordNum
  }

  override def apply(i: Int): ByteRecord = {
    ByteRecord(imgs.get(i), label(i.toInt)) // TODO: we may change this to Long
  }
}


/**
 * Wrap a RDD as a DataSet.
 * @param buffer
 */
// T is the returning value type. like ByteRecord
class AEPCachedDataSet[A: ClassTag, T: ClassTag]
(buffer: RDD[ArrayLike[A]], converter: A => T)
  extends DistributedDataSet[T] {

  protected lazy val count: Long = buffer.mapPartitions(iter => {
    require(iter.hasNext)
    val array = iter.next()
    require(!iter.hasNext)
    Iterator.single(array.length)
  }).reduce(_ + _)

  protected var indexes: RDD[Array[Int]] = buffer.mapPartitions(iter => {
    Iterator.single[Array[Int]]((0 until iter.next().length.toInt).toArray[Int])
  }).setName("original index").cache()


  override def data(train: Boolean): RDD[T] = {
    val _train = train
    val _converter = converter
    buffer.zipPartitions(indexes)((dataIter, indexIter) => {
      val indexes = indexIter.next()
      val indexOffset = math.max(1, indexes.length)
      val localData = dataIter.next()
      val offset = if (_train) {
        RandomGenerator.RNG.uniform(0, indexOffset).toInt
      } else {
        0
      }
      new Iterator[T] {
        private val _offset = new AtomicInteger(offset)

        override def hasNext: Boolean = {
          if (_train) true else _offset.get() < localData.length
        }

        override def next(): T = {
          val i = _offset.getAndIncrement()
          if (_train) {
            // indexes is an Array, we should improve this
            // meaning the maximum value is limited by Int
            _converter(localData(indexes(i % localData.length)))
          } else {
            if (i < localData.length) {
              _converter(localData(indexes(i)))
            } else {
              null.asInstanceOf[T]
            }
          }
        }
      }
    })
  }

  override def size(): Long = count

  override def shuffle(): Unit = {
      indexes.unpersist()
      indexes = buffer.mapPartitions(iter => {
        Iterator.single(RandomGenerator.shuffle((0 until iter.next().length).toArray))
      }).setName("shuffled index").cache()
  }

  override def originRDD(): RDD[_] = buffer

  override def cache(): Unit = {
    buffer.count()
    indexes.count()
    isCached = true
  }

  override def unpersist(): Unit = {
    buffer.unpersist()
    indexes.unpersist()
    isCached = false
  }
}


class AEPImageSet(var rdd: RDD[ByteRecord]) extends ImageSet {
  // RDD[ByteRecord] ==cached=> RDD[(AEPBytesRecord)] => RDD[ImageFeature]

  private val rddOfByteRecordWithCache = ImageSet.cacheWithAEP(rdd).data(train = false)

  private val rddOfImageFeature = rddOfByteRecordWithCache.map{ rec => ImageFeature(rec.data, rec
    .label)}

  override def transform(transformer: Preprocessing[ImageFeature, ImageFeature]): ImageSet = {
    transformer(rddOfImageFeature)
    this
  }

  override def isLocal(): Boolean = false

  override def isDistributed(): Boolean = true

  override def toImageFrame(): ImageFrame = {
    ImageFrame.rdd(rddOfImageFeature)
  }

  override def toDataSet(): DataSet[ImageFeature] = {
     DataSet.rdd[ImageFeature](rddOfImageFeature)
    // TODO: cache with AEP here. and should not use DataSet.rdd
  }
}

object ImageSet {

  /**
   * create LocalImageSet
   * @param data array of ImageFeature
   */
  def array(data: Array[ImageFeature]): LocalImageSet = {
    new LocalImageSet(data)
  }

  /**
   * create LocalImageSet from array of bytes
   * @param data nested array of bytes, expect inner array is a image
   */
  def array(data: Array[Array[Byte]]): LocalImageSet = {
    val images = data.map(ImageFeature(_))
    ImageSet.array(images)
  }

  /**
   * create DistributedImageSet
   * @param data rdd of ImageFeature
   */
  def rdd(data: RDD[ImageFeature]): DistributedImageSet = {
    new DistributedImageSet(data)
  }
//
//  def cacheWithBlockManager[D](data: RDD[BlockId]): AEPCachedDataSet[BlockId, D] = {
//    val nodeNumber = EngineRef.getNodeNumber()
//    val coaleasedRdd = data.coalesce(nodeNumber, true)
//    val countPerPartition = coaleasedRdd.mapPartitions{iter =>
//      require(iter.hasNext)
//      val byteRecord = iter.next()
//      // iter.next() has consumed an item, so we need to add 1 here.
//      Iterator.single(iter.size + 1)}
//    val result = coaleasedRdd.zipPartitions(countPerPartition){(dataIter, countIter) => {
//      val count = countIter.next()
//      var i = 0
//      val result = ArrayBuffer[BlockId]()
//      // Array[blockid]
//      while(dataIter.hasNext) {
//        val data = dataIter.next()
//        val blockId = SparkExtension.getLocalBlockId(i + "") // TODO: blockId should be uniqe across
//        // cluster
//        BlockManagerWrapper.putSingle(blockId, data, StorageLevel
//          .MEMORY_AND_DISK)
//        i += 1
//      }
//      Iterator.single(new BlockIdArray(result.toArray))
//    }}.setName("cached AEP images")
//      .cache()
//
//    new AEPCachedDataSet[BlockId, D](result.asInstanceOf[RDD[ArrayLike[BlockId]]],
//      (x: BlockId) => {
//
//      })
//  }


  def cacheWithAEP(data: RDD[ByteRecord]): AEPCachedDataSet[ByteRecord, ByteRecord] = {
    val nodeNumber = EngineRef.getNodeNumber()
    val coaleasedRdd = data.coalesce(nodeNumber, true)
    val countPerPartition = coaleasedRdd.mapPartitions{iter =>
      require(iter.hasNext)
      var totalBytes: Long = 0L
      var totalRecordNum = 0
      while(iter.hasNext) {
        val byteRecord = iter.next()
        totalRecordNum += 1
        totalBytes += byteRecord.data.length
      }
      Iterator.single((totalRecordNum, totalBytes))
    }
    val result = coaleasedRdd.zipPartitions(countPerPartition){(dataIter, countIter) => {
      val count = countIter.next()
      val aepArray = new AEPVarBytesArray(count._1, count._2)
      val labelBuffer = ArrayBuffer[Float]()
      var i = 0
      while(dataIter.hasNext) {
        val data = dataIter.next()
        val imgBuffer = ByteBuffer.wrap(data.data)
        val width = imgBuffer.getInt
        val height = imgBuffer.getInt
        aepArray.set(i, data.data)
        labelBuffer.append(data.label)
        i += 1
      }
      Iterator.single(AEPImageArray(aepArray, labelBuffer.toArray))
      }}.setName("cached AEP images")
      .cache()

    new AEPCachedDataSet[ByteRecord, ByteRecord](result.asInstanceOf[RDD[ArrayLike[ByteRecord]]],
      (x: ByteRecord) => x)
  }


  /**
   * create DistributedImageSet for a RDD of array bytes
   * @param data rdd of array of bytes
   */
  def rddBytes(data: RDD[Array[Byte]]): DistributedImageSet = {
    val images = data.map(ImageFeature(_))
    ImageSet.rdd(images)
  }

  /**
   * Read images as Image Set
   * if sc is defined, Read image as DistributedImageSet from local file system or HDFS
   * if sc is null, Read image as LocalImageSet from local file system
   *
   * @param path path to read images
   * if sc is defined, path can be local or HDFS. Wildcard character are supported.
   * if sc is null, path is local directory/image file/image file with wildcard character
   * @param sc SparkContext
   * @param minPartitions A suggestion value of the minimal partition number
   * @param resizeH height after resize, by default is -1 which will not resize the image
   * @param resizeW width after resize, by default is -1 which will not resize the image
   * @param imageCodec specifying the color type of a loaded image, same as in OpenCV.imread.
   *              By default is Imgcodecs.CV_LOAD_IMAGE_UNCHANGED
   * @return ImageSet
   */
  def read(path: String, sc: SparkContext = null, minPartitions: Int = 1,
           resizeH: Int = -1, resizeW: Int = -1,
           imageCodec: Int = Imgcodecs.CV_LOAD_IMAGE_UNCHANGED): ImageSet = {
    val imageSet = if (null != sc) {
      val images = sc.binaryFiles(path, minPartitions).map { case (p, stream) =>
          ImageFeature(stream.toArray(), uri = p)
      }
      ImageSet.rdd(images)
    } else {
      val files = Utils.listLocalFiles(path)
      val images = files.map { p =>
        ImageFeature(FileUtils.readFileToByteArray(p), uri = p.getAbsolutePath)
      }
      ImageSet.array(images)
    }
    if (resizeW == -1 || resizeH == -1) {
      imageSet -> ImageBytesToMat(imageCodec = imageCodec)
    } else {
      imageSet -> BufferedImageResize(resizeH, resizeW) -> ImageBytesToMat(imageCodec = imageCodec)
    }
  }

  /**
   * Convert ImageFrame to ImageSet
   *
   * @param imageFrame imageFrame which needs to covert to Imageset
   * @return ImageSet
   */
  private[zoo] def fromImageFrame(imageFrame: ImageFrame): ImageSet = {
    val imageset = imageFrame match {
      case distributedImageFrame: DistributedImageFrame =>
        ImageSet.rdd(imageFrame.toDistributed().rdd)
      case localImageFrame: LocalImageFrame =>
        ImageSet.array(imageFrame.toLocal().array)
    }
    imageset
  }
}
