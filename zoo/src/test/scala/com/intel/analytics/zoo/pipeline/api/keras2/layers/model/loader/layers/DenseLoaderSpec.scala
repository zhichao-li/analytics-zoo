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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.layers.{InputLayer, KerasRunner}
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras2.layers.Keras2ModelLoadingBaseSpec
import com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader.LayerLoader

class DenseLoaderSpec extends Keras2ModelLoadingBaseSpec {
  "Dense" should "be the same as Keras" in {
    val modelPath = randomModelPath()
    val kerasCode =
      s"""
        |input_tensor = Input(shape=[3])
        |input = np.random.uniform(0, 1, [1, 3])
        |output_tensor = Dense(2, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer='l1', activation="relu")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
        |from keras.models import load_model
        |model.save("${modelPath}")
      """.stripMargin
    KerasRunner.run(kerasCode)
    val seq = Sequential[Float]()
    val input = InputLayer[Float](inputShape = Shape(3), name = "input1")
    seq.add(input)
    val dense = reloadOutputLayerForModel(modelPath)
    seq.add(dense)
    LayerLoader.setWeights(dense, modelPath)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter = DenseLoader.toZooFormat, resetWeights = false)
  }
}

