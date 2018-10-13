package com.intel.analytics.zoo.aep;

public class AEPSimulator {
    static {
        try {
            System.loadLibrary("aep");
        } catch (UnsatisfiedLinkError e) {
            throw e;
        }
    }

    private static native void initAEP(String path, long size);

    public static native long allocate(long size);

    public static native void free(long address);

    public static native float get(long address, int offset);

}

// lizhichao@ ~/god/analytics-zoo/zoo/target/classes %>javah -d /Users/lizhichao/god/analytics-zoo/zoo/src/main/java/com/intel/analytics/zoo/aep com.intel.analytics.zoo.aep.AEPSimulator
