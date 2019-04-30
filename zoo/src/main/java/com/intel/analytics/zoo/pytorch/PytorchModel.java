package com.intel.analytics.zoo.pytorch;
import com.intel.analytics.zoo.pipeline.inference.JTensor;


public class PytorchModel {
    public native void load(String path);

    public native JTensor forward(float[] storage, int offset, int[] shape);
}
