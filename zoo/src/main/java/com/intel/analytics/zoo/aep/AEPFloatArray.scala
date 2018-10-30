
package com.intel.analytics.zoo.aep

import org.apache.spark.unsafe.Platform

object AEPFloatArray {

  def getTotalBytes(size: Long): Long = {
    size * 4
  }
  val MOVE_STEPS = 4

  def apply(iterator: Iterator[Float], numOfRecord: Int): AEPFloatArray = {
    val aepArray = new AEPFloatArray(numOfRecord)
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
class AEPFloatArray(val recordNum: Int) extends AEPArray[Float](
  recordNum * AEPFloatArray.MOVE_STEPS) {

  override  def get(i: Int): Float = {
    assert(!deleted)
    Platform.getFloat(null, indexOf(i))
  }

  def getMoveSteps(): Int = AEPFloatArray.MOVE_STEPS

  def set(i: Int, value: Float): Unit = {
    assert(!deleted)
    Platform.putFloat(null, indexOf(i), value)
  }

  protected def indexOf(i: Int): Long = {
    val index = startAddr + (i * AEPFloatArray.MOVE_STEPS)
    assert(index <= lastOffSet)
    index
  }
}
