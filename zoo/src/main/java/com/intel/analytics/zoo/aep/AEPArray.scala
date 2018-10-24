package com.intel.analytics.zoo.aep

import org.apache.spark.unsafe.Platform

/**
 *
 * @param startAddr the starting address of this array
 * @param size the item size of this array
 */
abstract class AEPArray[T](startAddr: Long, size: Long) {

  val totalBytes: Long = size * getMoveSteps()
  assert(totalBytes > 0, "The size of bytes should be larger than 0!")

  val lastOffSet = startAddr + totalBytes

  var deleted: Boolean = false

  def getMoveSteps(): Int

  def get(i: Long): T

  def set(i: Long, value: T): Unit

  def free(): Unit = {
    if (!deleted) {
      AEPHandler.free(startAddr)
      deleted = true
    }
  }

  protected def indexOf(i: Long): Long = {
    val index = startAddr + (i * getMoveSteps())
    assert(index <= lastOffSet)
    index
  }
}
