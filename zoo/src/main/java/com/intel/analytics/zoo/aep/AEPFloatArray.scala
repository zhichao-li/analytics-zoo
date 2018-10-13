
package com.intel.analytics.zoo.aep

import org.apache.spark.unsafe.Platform

/**
  * An float array with fixed size stored in AEP.
 *  @param startAddr the start address of the array
  * @param size the size of the array
  */
class AEPFloatArray(val startAddr: Long, val size: Long) {

  def AEPFloatArray(size: Long) {
    val startAddr: Long = Platform.allocateMemory(totalBytes)
    assert(startAddr > 0, "Not enough memory!")
    new AEPFloatArray(startAddr, size)
  }
  val MOVE_STEPS = 2
  val totalBytes: Long = size << MOVE_STEPS
  assert(totalBytes > 0, "The size of bytes should be larger than 0!")

  val lastOffSet = startAddr + totalBytes

  var deleted: Boolean = false

  def get(i: Long): Double = {
    assert(!deleted)
    Platform.getDouble(null, indexOf(i))
  }

  def set(i: Long, value: Double): Unit = {
    assert(!deleted)
    Platform.putDouble(null, indexOf(i), value)
  }

  def free(): Unit = {
    if (!deleted) {
      AEPSimulator.free(startAddr)
      deleted = true
    }
  }

  private def indexOf(i: Long): Long = {
    val index = startAddr + (i << MOVE_STEPS)
    assert(index < lastOffSet)
    index
  }
}
