package com.intel.analytics.zoo.persistent.memory;

public class PersistentMemoryAllocator {

    static {
        try {
            System.loadLibrary("persistent_memory_allocator");
        } catch (UnsatisfiedLinkError e) {
            throw e;
        }
        String memPath = "/mnt/pmem0"; // this value should be read from config or vm parameter
        long sizeByByte = 248 * 1024 * 1024 * 1024;
        initialize(memPath, sizeByByte);
    }

    private static native void initialize(String path, long size);

    public static native long allocate(long size);

    public static native void free(long address);

    public static native void copy(long destAddress, long srcAddress, long size);
}
