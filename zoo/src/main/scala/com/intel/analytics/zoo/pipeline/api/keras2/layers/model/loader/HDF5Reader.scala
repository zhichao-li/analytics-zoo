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

import com.intel.analytics.bigdl.tensor.Tensor
import org.bytedeco.javacpp.hdf5._
import org.bytedeco.javacpp.{BytePointer, FloatPointer, hdf5}


object HDF5Reader {
  def apply(hdf5File: String): HDF5Reader = {
    new HDF5Reader(hdf5File)
  }
}

class HDF5Reader(hdf5File: String) {
  val hdf5FileInstance = new H5File(hdf5File, hdf5.H5F_ACC_RDONLY)
  val dataType = new hdf5.DataType(hdf5.PredType.NATIVE_FLOAT())
  /**
   * Read data from groupPath
   * @param dataSetPath path to the specify DataSet
   * @return A Tensor[Float] for the given DataSet
   */
  def readDataSet(dataSetPath: String): Tensor[Float] = {
    val paths = dataSetPath.split("/").filter(!_.isEmpty)
    val datasetName = paths.last
    val groupInstance = getGroup(paths.slice(0, paths.length - 1))
    val data = readDataSet(groupInstance, datasetName)
    groupInstance.deallocate()
    return data
  }

  /**
   * Read string attribute by name.
   *
   * @param attributePath Name of attribute
   * @param attributeLength  Length of the attribute
   */
  def readAttribute(attributePath: String, attributeLength: Int = 2000000): String = {
    val items = attributePath.split("/").filter(!_.isEmpty)
    val group = if (items.length == 1) {
      this.hdf5FileInstance
    } else {
      this.getGroup(items.slice(0, items.length - 1))
    }
    val attribute = items.last
    val attributeIns = group.openAttribute(attribute)
    val result = readAttribute(attributeIns, attributeLength)
    attributeIns.deallocate()
    if (result.endsWith("\u0000")) {
      result.replace("\u0000", "")
    } else {
      result
    }
  }

  private def readDataSet(fileGroup: Group, datasetName: String): Tensor[Float] = {
    val dataset = fileGroup.openDataSet(datasetName)
    val space = dataset.getSpace()
    val dims = new Array[Long](space.getSimpleExtentNdims())
    space.getSimpleExtentDims(dims)
    val storage = new Array[Float](dims.product.toInt)
    val fp = new FloatPointer(storage: _*)
    dataset.read(fp, dataType)
    fp.get(storage)
    space.deallocate()
    dataset.deallocate()
    return Tensor(data = storage, shape = dims.map(_.toInt))
  }

  // You need to manually deallocate the returning value
  private def getGroup(groupPath: Array[String]): hdf5.Group = {
    println(groupPath.mkString("/"))
    if (groupPath.isEmpty) {
      return this.hdf5FileInstance
    }
    var groupTmp = this.hdf5FileInstance.openGroup(groupPath(0))
    var group = groupTmp
    var i = 1
    while (i < groupPath.length) {
      group = groupTmp.openGroup(groupPath(i))
      groupTmp.deallocate()
      groupTmp = group
      i += 1
    }
    group
  }

  private def readAttribute(attribute: hdf5.Attribute, bufferSize: Int): String = {
    val varLenType = attribute.getVarLenType()
    val attrBuffer: Array[Byte] = new Array[Byte](bufferSize)
    val attrPointer = new BytePointer(attrBuffer: _*)
    attribute.read(varLenType, attrPointer)
    attrPointer.get(attrBuffer)
    varLenType.deallocate()
    new String(attrBuffer)
  }

}
