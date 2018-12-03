
package com.intel.analytics.zoo.persistent.memory

import org.apache.spark.unsafe.Platform

object OptaneDCFloatArray {

  def getTotalBytes(size: Long): Long = {
    size * 4
  }
  val MOVE_STEPS = 4

  def apply(iterator: Iterator[Float], numOfRecord: Int): OffHeapFloatArray = {
    val aepArray = new OffHeapFloatArray(numOfRecord)
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
  * @param recordNum number of item for this array.
  */
class OffHeapFloatArray(val recordNum: Int, memoryType: MemoryType = OptaneDC) extends OffHeapArray[Float](
  recordNum * OptaneDCFloatArray.MOVE_STEPS, memoryType) {

  override  def get(i: Int): Float = {
    assert(!deleted)
    Platform.getFloat(null, indexOf(i))
  }

  def getMoveSteps(): Int = OptaneDCFloatArray.MOVE_STEPS

  def set(i: Int, value: Float): Unit = {
    assert(!deleted)
    Platform.putFloat(null, indexOf(i), value)
  }

  protected def indexOf(i: Int): Long = {
    val index = startAddr + (i * OptaneDCFloatArray.MOVE_STEPS)
    assert(index <= lastOffSet)
    index
  }
}
