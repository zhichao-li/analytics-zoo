package com.intel.analytics.zoo.persistent.memory

import org.apache.spark.SparkEnv

object PersistentMemoryStore {
  // TODO: Maybe we can use docker to combine the numa socket and persistent memory.
  // TODO: Passing the path as a parameter via sparkconf?
  val memPaths = List("/dev/dax0.0", "/dev/dax1.0")
  val memSizePerByte = 248 * 1024 * 1024 * 1024
  // TODO: There might be problem if we have executorID 2 and 4 in the same machine
  val pathIndex = executorID % memPaths.length
  println(s"Executor: ${executorID()} is using ${memPaths(pathIndex)}")
  PersistentMemoryAllocator.initialize(memPaths(pathIndex), memSizePerByte)

  private def executorID(): Int = {
    if (SparkEnv.get.executorId.equals("driver")) {
      1
    } else {
      SparkEnv.get.executorId.toInt
    }
  }

  def allocate(size: Long): Long = {
    PersistentMemoryAllocator.allocate(size)
  }

  def free(address: Long): Unit = {
    PersistentMemoryAllocator.free(address)
  }

  def copy(destAddress: Long, srcAddress: Long, size: Long): Unit = {
    PersistentMemoryAllocator.copy(destAddress, srcAddress, size)
  }

}
