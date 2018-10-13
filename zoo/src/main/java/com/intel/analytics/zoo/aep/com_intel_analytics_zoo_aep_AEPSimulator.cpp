#include <cstring>

#include "com_intel_analytics_zoo_aep_AEPSimulator.h"

/*
 * Class:     com_intel_analytics_zoo_aep_AEPSimulator
 * Method:    initAEP
 * Signature: (Ljava/lang/String;J)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_aep_AEPSimulator_initAEP
  (JNIEnv *, jclass, jstring, jlong) {
  }

/*
 * Class:     com_intel_analytics_zoo_aep_AEPSimulator
 * Method:    allocate
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_analytics_zoo_aep_AEPSimulator_allocate
  (JNIEnv *, jclass, jlong) {

  float *addr = new float(jlong);
  return addr;
  }

/*
 * Class:     com_intel_analytics_zoo_aep_AEPSimulator
 * Method:    free
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_analytics_zoo_aep_AEPSimulator_free
  (JNIEnv *, jclass, jlong) {
     delete jlong;
  }

/*
 * Class:     com_intel_analytics_zoo_aep_AEPSimulator
 * Method:    get
 * Signature: (JI)F
 */
JNIEXPORT jfloat JNICALL Java_com_intel_analytics_zoo_aep_AEPSimulator_get
  (JNIEnv * env, jclass jcl, jlong address, jint offset) {
  return address + offset * 4 *

}

// what's the pointer type return by aep? and how to access the element by that address?
// how to write the data into that buffer?
// 拿到一个float array之后，如何写入到预先分配到的buffer中去？
