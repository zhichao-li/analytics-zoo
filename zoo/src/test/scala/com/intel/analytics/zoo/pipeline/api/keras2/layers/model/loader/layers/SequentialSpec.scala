/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader.layers

import java.util.regex.Pattern

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.{InputLayer, KerasRunner}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Keras2ModelLoadingBaseSpec
import com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader.{LayerLoader, ModelLoader}

class SequentialSpec extends Keras2ModelLoadingBaseSpec {
  "Sequential" should "be the same as Keras" in {

//    val str = "dense_1/kernel:0dense_1/bias:0"
//
//    val attributeMatcher = Pattern.compile(":\\d+").matcher(str);
//    val foundTfGroups = attributeMatcher.find();
//    foundTfGroups
    val modelPath = randomModelPath()
    val kerasCode =
      s"""
         |model = Sequential()
         |dense = Dense(3, input_shape=[2])
         |model.add(dense)
         |input = np.random.uniform(0, 1, [1, 2])
         |from keras.models import load_model
         |model.save("${modelPath}")
      """.stripMargin
    KerasRunner.run(kerasCode)
    val model = ModelLoader.load(modelPath)
    checkOutputAndGrad(model.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter = DenseLoader.toZooFormat, resetWeights = false)
  }
}

