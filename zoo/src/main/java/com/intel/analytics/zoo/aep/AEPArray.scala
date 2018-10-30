package com.intel.analytics.zoo.aep

import org.apache.spark.unsafe.Platform

/**
 *
 * @param totalBytes
 */
abstract class AEPArray[T](totalBytes: Long) {

  //        val startAddr = AEPHandler.allocate(recordNumber * recordBytes)
  val startAddr: Long = Platform.allocateMemory(totalBytes)
  assert(startAddr > 0, "Not enough memory!")
  assert(totalBytes > 0, "The size of bytes should be larger than 0!")

  val lastOffSet = startAddr + totalBytes

  var deleted: Boolean = false

  def get(i: Int): T

  def set(i: Int, value: T): Unit

  def free(): Unit = {
    if (!deleted) {
      AEPHandler.free(startAddr)
      deleted = true
    }
  }

  protected def indexOf(i: Int): Long
}
