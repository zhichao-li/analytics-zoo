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
