/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_intel_analytics_zoo_pipeline_api_net_PytorchModel */

#ifndef _Included_com_intel_analytics_zoo_pipeline_api_net_PytorchModel
#define _Included_com_intel_analytics_zoo_pipeline_api_net_PytorchModel
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    loadModelNative
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_loadModelNative
  (JNIEnv *, jclass, jstring);

/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    saveModelNative
 * Signature: (JLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_saveModelNative
  (JNIEnv *, jclass, jlong, jstring);

/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    loadLossNative
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_loadLossNative
  (JNIEnv *, jclass, jstring);

/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    modelForwardNative
 * Signature: (JZ[[F[I[[I)[Lcom/intel/analytics/zoo/pipeline/inference/JTensor;
 */
JNIEXPORT jobjectArray JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_modelForwardNative
  (JNIEnv *, jclass, jlong, jboolean, jobjectArray, jintArray, jobjectArray);

/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    modelBackwardNative
 * Signature: (J[[F[I[[I)[Lcom/intel/analytics/zoo/pipeline/inference/JTensor;
 */
JNIEXPORT jobjectArray JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_modelBackwardNative
  (JNIEnv *, jclass, jlong, jobjectArray, jintArray, jobjectArray);

/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    lossForwardNative
 * Signature: (J[[F[I[[I[[F[I[[I)Lcom/intel/analytics/zoo/pipeline/inference/JTensor;
 */
JNIEXPORT jobject JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_lossForwardNative
  (JNIEnv *, jclass, jlong, jobjectArray, jintArray, jobjectArray, jobjectArray, jintArray, jobjectArray);

/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    lossBackwardNative
 * Signature: (J)[Lcom/intel/analytics/zoo/pipeline/inference/JTensor;
 */
JNIEXPORT jobjectArray JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_lossBackwardNative
  (JNIEnv *, jclass, jlong);

/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    getGradientNative
 * Signature: (J)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_getGradientNative
  (JNIEnv *, jclass, jlong);

/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    updateWeightNative
 * Signature: (J[F)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_updateWeightNative
  (JNIEnv *, jclass, jlong, jfloatArray);

/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    getWeightNative
 * Signature: (J)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_getWeightNative
  (JNIEnv *, jclass, jlong);

/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    releaseModelNative
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_releaseModelNative
  (JNIEnv *, jclass, jlong);

/*
 * Class:     com_intel_analytics_zoo_pipeline_api_net_PytorchModel
 * Method:    releaseLossNative
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_pipeline_api_net_PytorchModel_releaseLossNative
  (JNIEnv *, jclass, jlong);

#ifdef __cplusplus
}
#endif
#endif
