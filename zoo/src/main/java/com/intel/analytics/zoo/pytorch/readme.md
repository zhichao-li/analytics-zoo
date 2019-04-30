# Build dynamic lib

cd /home/lizhichao/bin/god/analytics-zoo/zoo/target/classes
javah -d /home/lizhichao/bin/god/analytics-zoo/zoo/src/main/java/com/intel/analytics/zoo/pytorch/native/  com.intel.analytics.zoo.pytorch.PytorchModel

cd /home/lizhichao/bin/god/analytics-zoo/zoo/src/main/java/com/intel/analytics/zoo/pytorch/native/build
cmake -DCMAKE_PREFIX_PATH=/opt/work/ptorch/libtorch/ .. 

# Driver class for test JNI interface

Test.java would load a exported model, and `forward` with dummy input data.

export LD_LIBRARY_PATH=/opt/work/analytics-zoo/zoo/src/main/java/com/intel/analytics/zoo/pytorch/native/build/
/home/lizhichao/bin/god/analytics-zoo/zoo/src/main/java/com/intel/analytics/zoo/pytorch/Test.java
