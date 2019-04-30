package com.intel.analytics.zoo.pytorch;
import com.intel.analytics.zoo.pipeline.inference.JTensor;

public class Test {

    static {
        try {
            System.loadLibrary("pytorch-engine");
            //System.load("/opt/work/analytics-zoo/zoo/src/main/java/com/intel/analytics/zoo/pytorch/native/build/libpytorch-engine.so");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load.\n" + e);
            System.exit(1);
        }
    }

    public static void main(String argv[])
    {
        PytorchModel pmodel = new PytorchModel();
        String model = "/opt/work/model.pt";        
        pmodel.load(model);
        float[] storage = new float[1*3*224*224];
        int i = 0;
        for (;i < storage.length;i++){
            storage[i] = 1;
      
        }
        int[] shape = {1, 3, 224, 224};
        JTensor result = pmodel.forward(storage, 0, shape);
        System.out.println(result);
    }
}
