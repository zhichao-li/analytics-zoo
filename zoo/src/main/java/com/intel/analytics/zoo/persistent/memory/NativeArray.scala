package com.intel.analytics.zoo.persistent.memory

import org.apache.spark.unsafe.Platform

sealed trait MemoryType

case object OptaneDC extends MemoryType

case object DRAM extends MemoryType

/**
 *
 * @param totalBytes
 */
abstract class OffHeapArray[T](totalBytes: Long, memoryType: MemoryType) {

  val startAddr: Long = if (memoryType == OptaneDC) {
    println("Using persistent memory")
    PersistentMemoryStore.allocate(totalBytes)
  } else {
    println("Using main memory")
    Platform.allocateMemory(totalBytes)
  }

  assert(startAddr > 0, "Not enough memory!")
  assert(totalBytes > 0, "The size of bytes should be larger than 0!")

  val lastOffSet = startAddr + totalBytes

  var deleted: Boolean = false

  def get(i: Int): T

  def set(i: Int, value: T): Unit

  def free(): Unit = {
    if (!deleted) {
      if (memoryType == OptaneDC) {
        PersistentMemoryAllocator.free(startAddr)
      } else {
        Platform.freeMemory(startAddr)
      }
      deleted = true
    }
  }

  protected def indexOf(i: Int): Long
}
