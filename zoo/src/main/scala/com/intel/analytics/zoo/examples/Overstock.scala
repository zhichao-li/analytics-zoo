package com.intel.analytics.zoo.examples

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, _}
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, Shape}
import com.intel.analytics.zoo.pipeline.api.autograd.{CustomLoss, Variable, AutoGrad => A}
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.{KerasNet, Sequential}

class Overstock {

}

// we need a evaluate method for debug purpose.
object Overstock {

  // yTrue: (batch, 1), yPred: (batch, 1)
  def cLoss(yTrue: Variable[Float], yPred: Variable[Float])(
      implicit ev: TensorNumeric[Float]): Variable[Float] = {
    val reOrderRate = 0.045
    val lostOrders = 0.5
    val juiceAdj = 0.18
    val cancelRatio = 0.21
    val leakageRate = 0.0123
    val chargebackRecovery = 0.18
    val labor = 2.05
    val binaryPred = BinaryThreshold[Float](0.5).from(yPred)
    val truePositive = A.sum(yTrue * binaryPred, axis = 0, keepdims = true)
    val falsePositive = A.sum(BinaryThreshold[Float](0.5).from(yTrue - yPred), axis = 0, keepdims = true)
    val falseNegative = A.sum(BinaryThreshold[Float](0.5).from(yPred - yTrue), axis = 0, keepdims = true)

    val FNLoss = falseNegative * (1 - chargebackRecovery)
    val TPLoss = truePositive * leakageRate + labor
    val FPLoss = falsePositive * reOrderRate * lostOrders * juiceAdj * cancelRatio + labor
    val TNLoss = 0

    return FNLoss + TPLoss + FPLoss + TNLoss
  }

  val customLoss = CustomLoss[Float](cLoss _, sizeAverage = true)

  def lr(): KerasNet[Float] = {
    val seq = Sequential[Float]()
    seq.add(Dense[Float](1, inputShape = Shape(5)))
    seq.add(Activation[Float]("sigmoid"))
  }

  def toBatch(features: Array[Tensor[Float]],
      labels: Array[Tensor[Float]]): LocalDataSet[MiniBatch[Float]] = {
    val samples = features.zip(labels).map(pair => Sample(pair._1, pair._2))
    val batchSample = SampleToMiniBatch[Float](batchSize = 4).apply(samples.iterator)
    DataSet.array(batchSample.toArray)
  }

  def testModel(): Unit = {
    System.setProperty("bigdl.localMode", "true")
    System.setProperty("bigdl.coreNumber", "1")
    Engine.init
    val model = lr()
    //    model.compile(optimizer = new SGD[Float](learningRate = 0.001), BCECriterion[Float]())
    model.compile(optimizer = new SGD[Float](learningRate = 0.001),
      loss = customLoss.asInstanceOf[Criterion[Float]])
    val mockInputs = Range(0, 1000).map(_ => Tensor[Float](5).randn())
    val mockLabels = Range(0, 1000).map(_ => Tensor[Float](1).fill(1f))
    model.fit(toBatch(mockInputs.toArray, mockLabels.toArray), nbEpoch = 2)
  }

  def testLoss(): Unit = {
    val yTrue = Tensor[Float](data = Array(0f, 1f, 1f, 0f), shape = Array(4, 1))
    val yPred = Tensor[Float](data = Array(0.3f, 0.7f, 0.6f, 0.8f), shape = Array(4, 1))
    val loss = customLoss.forward(yPred, yTrue) // NOTE: the order is yPred then yTrue in forward
    println(loss)
  }

  def main(args: Array[String]): Unit = {
    //    testModel()
    testLoss()
  }
}
