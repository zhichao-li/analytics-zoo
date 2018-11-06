package com.intel.analytics.zoo.aep;

public class AEPHandler {
    static {
        try {
            System.loadLibrary("aep");
        } catch (UnsatisfiedLinkError e) {
            throw e;
        }
    }

    public static native void initialize(String path, long size);

    public static native long allocate(long size);

    public static native void free(long address);

    public static native void copy(long destAddress, long srcAddress, long size);


}

// lizhichao@ ~/bin/god/zoo/zoo/target/classes %>javah -d /Users/lizhichao/god/analytics-zoo/zoo/src/main/java/com/intel/analytics/zoo/aep com.intel.analytics.zoo.aep.AEPHandler

// what's the pointer type return by aep? and how to access the element by that address?
// how to write the data into that buffer?
// 拿到一个float array之后，如何写入到预先分配到的buffer中去？
