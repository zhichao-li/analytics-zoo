package com.intel.analytics.zoo.pipeline.api.model.loader

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.FloatPointer
import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.hdf5._
import org.bytedeco.javacpp.hdf5
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.FloatPointer
import org.bytedeco.javacpp.Loader
import scala.collection.JavaConverters._



object HDF5Reader {

  def main(args: Array[String]) {
//    HDF5Reader("/tmp/")
  }
}

class HDF5Reader(hdf5File: String) {
  val hdf5FileInstance = new H5File(hdf5File, hdf5.H5F_ACC_RDONLY)
  val dataType = new hdf5.DataType(hdf5.PredType.NATIVE_FLOAT())

  /**
   * Read data set as ND4J array from HDF5 group.
   *
   * @param fileGroup   HDF5 file or group
   * @param datasetName Name of data set
   * @return INDArray from HDF5 data set
   */
  private def readDataSet(fileGroup: Group, datasetName: String){
//    val fileGroup: hdf5.Group = this.hdf5FileInstance
    val dataset = fileGroup.openDataSet(datasetName);
    val space = dataset.getSpace()
    val nbDims = space.getSimpleExtentNdims()
    val dims = Array[Long](nbDims) //new long[nbDims];
    space.getSimpleExtentDims(dims);
    var dataBuffer: Array[Float] = null
    var fp: FloatPointer = null
    val totalLength = 80 //dims.product
    dataBuffer = new Array[Float](totalLength.toInt)

    fp = new FloatPointer(dataBuffer: _*)
    dataset.read(fp, dataType)
    fp.get(dataBuffer);  // dataBuffer would storing the data
    space.deallocate();
    dataset.deallocate();
    return dataBuffer;

  }

  def readDataSet(datasetName: String, groups: Array[String]) {
    if (groups.length == 0)
      return readDataSet(this.hdf5FileInstance, datasetName)
    val groupArray = openGroups(groups)
    val data = readDataSet(groupArray(groupArray.length - 1), datasetName);
    closeGroups(groupArray);
    return data
  }

  def openGroups(groups: Array[String]): Array[Group] = {
    val groupArray = new Array[hdf5.Group](groups.length)
    groupArray(0) = this.hdf5FileInstance.openGroup(groups(0))
    var i = 1
    while (i < groups.length) {
      groupArray(i) = groupArray(i - 1).openGroup(groups(i))
      i += 1
    }
    groupArray
  }

  def closeGroups(groupArray: Array[hdf5.Group]) {
    groupArray.map{_.deallocate()}
  }


  /**
   * Read string attribute from group path.
   *
   * @param attributeName Name of attribute
   * @param bufferSize    buffer size to read
   * @return Fixed-length string read from HDF5 attribute name
   */
  def readAttributeAsFixedLengthString(attributeName: String, bufferSize: Int): String = {
    val a = hdf5FileInstance.openAttribute(attributeName);
    val s = readAttributeAsFixedLengthString(a, bufferSize);
    a.deallocate();
    return s;
  }

  /**
   * Read attribute of fixed buffer size as string.
   *
   * @param attribute HDF5 attribute to read as string.
   * @return Fixed-length string read from HDF5 attribute
   */
  private def readAttributeAsFixedLengthString(attribute: hdf5.Attribute, bufferSize: Int): String = {
    val vl = attribute.getVarLenType()
    val attrBuffer: Array[Byte] = new Array[Byte](bufferSize)
    val attrPointer = new BytePointer(attrBuffer: _*)
    attribute.read(vl, attrPointer)
    attrPointer.get(attrBuffer)
    vl.deallocate()
    new String(attrBuffer)
  }

}
