package com.intel.analytics.zoo.persistent.memory;

public class PersistentMemoryAllocator {

    static {
        new com.intel.analytics.zoo.persistent.memory.NativeLoader().init();
    }

    public static native void initialize(String path, long size);

    public static native long allocate(long size);

    public static native void free(long address);

    public static native void copy(long destAddress, long srcAddress, long size);
}
