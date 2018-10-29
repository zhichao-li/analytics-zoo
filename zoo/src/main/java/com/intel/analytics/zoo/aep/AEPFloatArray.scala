
package com.intel.analytics.zoo.aep

import org.apache.spark.unsafe.Platform

object AEPFloatArray {

  def getTotalBytes(size: Long): Long = {
    size * 4
  }
  val MOVE_STEPS = 4

  def apply(size: Long): AEPFloatArray = {
    val startAddr: Long = AEPHandler.allocate(AEPFloatArray.getTotalBytes(size))
    assert(startAddr > 0, "Not enough memory!")
    new AEPFloatArray(startAddr, size)
  }

  def apply(iterator: Iterator[Float], size: Long): AEPFloatArray = {
    val aepArray = AEPFloatArray(size)
    var i = 0
    while(iterator.hasNext) {
      aepArray.set(i, iterator.next())
      i += 1
    }
    aepArray
  }
}
/**
  * An float array with fixed size stored in AEP.
 *  @param startAddr the start address of the array
  * @param size number of item for this array.
  */
class AEPFloatArray(val startAddr: Long, val size: Long) extends AEPArray[Float](startAddr, size) {

  override  def get(i: Long): Float = {
    assert(!deleted)
    Platform.getFloat(null, indexOf(i))
  }

  def getMoveSteps(): Int = AEPFloatArray.MOVE_STEPS

  def set(i: Long, value: Float): Unit = {
    assert(!deleted)
    Platform.putFloat(null, indexOf(i), value)
  }
}
