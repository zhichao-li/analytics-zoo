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

package com.intel.analytics.zoo.pipeline.api

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{CAddTable, MM, SpatialCrossMapLRN, Sum}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.autograd.Variable
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.{KerasNet, Model => ZModel}

class NetSpec extends ZooSpecHelper{

  "invokeMethod set inputShape" should "work properly" in {
    KerasUtils.invokeMethod(Dense[Float](3), "_inputShapeValue_$eq", Shape(2, 3))
  }

  "Load Caffe model" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "caffe"
    val model = Net.loadCaffe[Float](
      s"$path/test_persist.prototxt", s"$path/test_persist.caffemodel")
    val newModel = model.newGraph("ip")
    newModel.outputNodes.head.element.getName() should be("ip")
  }

  "createTmpFile" should "work properly" in {
    val tmpFile = ZooSpecHelper.createTmpFile()
    print(tmpFile)
  }

  "Load Keras-style Analytics Zoo model" should "work properly" in {
    val input = Input[Float](inputShape = Shape(3, 5))
    val d = Dense[Float](7).setName("dense1").inputs(input)
    val model = ZModel[Float](input, d)

    val tmpFile = createTmpFile()
    val absPath = tmpFile.getAbsolutePath
    model.saveModule(absPath, overWrite = true)

    val reloadedModel = Net.load[Float](absPath)
      .asInstanceOf[KerasNet[Float]]

    val inputTensor = Tensor[Float](2, 3, 5).rand()
    compareOutputAndGradInput(
      model.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      reloadedModel.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]], inputTensor)
  }

  "Load BigDL model" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "bigdl"
    val model = Net.loadBigDL[Float](s"$path/bigdl_lenet.model")
    val newModel = model.newGraph("reshape2")
    newModel.outputNodes.head.element.getName() should be ("reshape2")
  }

  "Load Torch model" should "work properly" in {
    val layer = new SpatialCrossMapLRN[Float](5, 1.0, 0.75, 1.0)

    val tmpFile = java.io.File.createTempFile("module", ".t7")
    val absolutePath = tmpFile.getAbsolutePath
    layer.saveTorch(absolutePath, true)

    val reloadedModel = Net.loadTorch[Float](absolutePath)

    val inputTensor = Tensor[Float](16, 3, 224, 224).rand()
    compareOutputAndGradInput(
      layer.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      reloadedModel.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]], inputTensor)
  }

  "Load Tensorflow model" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "tensorflow"
    val model = Net.loadTF[Float](s"$path/lenet.pb", Seq("Placeholder"), Seq("LeNet/fc4/BiasAdd"))
    val newModel = model.newGraph("LeNet/fc3/Relu")
    newModel.outputNodes.head.element.getName() should be ("LeNet/fc3/Relu")
  }

  "net load model" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("models")
    val path = resource.getPath + "/" + "zoo_keras"

    val seq = Net.load[Float](s"$path/small_seq.model")
    seq.forward(Tensor[Float](2, 3, 5).rand())

    val model = Net.load[Float](s"$path/small_model.model")
    model.forward(Tensor[Float](2, 3, 5).rand())
  }

  "connect variable " should "work properly" in {
    def createOldModel(): AbstractModule[Activity, Activity, Float] = {
      val input1 = Input[Float](inputShape = Shape(3))
      val input2 = Input[Float](inputShape = Shape(3))
      val sum = new KerasLayerWrapper[Float](
        CAddTable[Float]().asInstanceOf[AbstractModule[Activity, Activity, Float]])
        .inputs(Array(input1, Dense[Float](3).inputs(input2)))
      ZModel[Float](input = Array(input1, input2), output = sum)
    }
    val input1 = Variable[Float](inputShape = Shape(3))
    val input2 = Variable[Float](inputShape = Shape(3))
    val diff = input1 + Dense[Float](3).from(input2)
    val model = ZModel[Float](input = Array(input1, input2), output = diff)
    val inputValue = Tensor[Float](1, 3).randn()
    val oldModel = createOldModel()
    val out = model.forward(T(inputValue, inputValue)).toTensor[Float]
    val oldOut = oldModel.forward(T(inputValue, inputValue)).toTensor[Float]
    out.almostEqual(oldOut, 1e-4)
  }

  "Load Tensorflow model from path" should "work properly" in {
    val resource = getClass().getClassLoader().getResource("tf")
    val model = Net.loadTF[Float](resource.getPath)
    val result = model.forward(Tensor[Float](4, 1, 28, 28).rand())
    result.toTensor[Float].size() should be (Array(4, 10))
  }

  "keras-like predict local multiple input" should "work properly" in {
    val input1 = Input[Float](inputShape = Shape(3))
    val input2 = Input[Float](inputShape = Shape(2))
    val out = Merge.merge(List(input1, input2), mode = "concat")
    val model = ZModel[Float](input = Array(input1, input2), output = out)
    val recordNum = 113
    val inputData = Array(Tensor[Float](recordNum, 3).rand(), Tensor[Float](recordNum, 2).rand())
    val result = model.predict(inputData, batch = 4)
    val fResult = model.forward(T.array(inputData))
    assert(result.almostEqual(fResult.toTensor[Float], 1e-5))
  }

  "keras-like predict local single output" should "work properly" in {
//    val input1 = Input[Float](inputShape = Shape(3))
//    val out = Dense[Float](4).inputs(input1)
//    val model = ZModel[Float](input = input1, output = out)
//    val recordNum = 113
//    val inputData = Tensor[Float](recordNum, 3).rand()
//    val result = model.predict(inputData, batch = 4)
//    val fResult = model.forward(inputData)
//    assert(result.almostEqual(fResult.toTensor[Float], 1e-5))
//    val i = Tensor[Float](recordNum, 2, 3).rand()
//    val i2 = Tensor[Float](recordNum, 2, 3).rand()
//    val mm = MM[Float](transB = true)
//    val o1 = mm.forward(T(i, i2))
//    mm.forward(T(i, i2))
//    mm.forward(T(i, i2))
//    mm.forward(T(i, i2))
//    val o2 = mm.forward(T(i, i2))
//    assert(o1.almostEqual(o2, 1e-5))

//    val sum = Sum[Float]()
//        val o1 = sum.forward(i)
//        val o2 = sum.forward(i)
//        assert(o1.almostEqual(o2, 1e-5))
//    import com.intel.analytics.zoo.pipeline.api.autograd.AutoGrad
//    val input1 = Variable[Float](inputShape = Shape(4, 2))
//    val input2 = Variable[Float](inputShape = Shape(4, 2))
//    val embedding1 = Embedding[Float](10, 20)
//    val embedding2 = Embedding[Float](10, 20)
//    val query = embedding1.from(input1)
//    val doc = embedding2.from(input2)
//    AutoGrad.dot(query, doc, axes = List(2, 2))

        import com.intel.analytics.zoo.pipeline.api.autograd.AutoGrad
        val input1 = Variable[Float](inputShape = Shape(4, 2))
        val input2 = Variable[Float](inputShape = Shape(4, 2))
        val result = AutoGrad.mm(input1, input2, axes = List(2, 2))
        val model = ZModel[Float](input = Array(input1, input2), output = result)
    val recordNum = 2
        val i1 = Tensor[Float](recordNum, 3, 4).rand()
        val i2 = Tensor[Float](recordNum, 3, 4).rand()
    val o1 = model.forward(T(i1, i2)).toTensor[Float].clone()
    val o2 = model.forward(T(i1, i2)).toTensor[Float].clone()
    assert(o1.almostEqual(o2, 1e-5))
  }
}
