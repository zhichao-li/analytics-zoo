package com.intel.analytics.zoo.examples

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad, CustomLoss, Variable}
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad => A}
import com.intel.analytics.zoo.pipeline.api.keras.layers._
import com.intel.analytics.zoo.pipeline.api.keras.models.{KerasNet, Sequential}

import scala.reflect.ClassTag

class Overstock {

}

// we need a evaluate method for debug purpose.
object Overstock {

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
    val truePositive = A.sum(yTrue * binaryPred)
    //val truePositive = Merge[Float](mode = "dot").from[Float](yTrue, binaryPred)
    val falsePositive = A.sum(BinaryThreshold[Float](0.5).from(yTrue - yPred))
    val falseNegative = A.sum(BinaryThreshold[Float](0.5).from(yPred - yTrue))

    val FNLoss = falseNegative * (1 - chargebackRecovery)
    val TPLoss = truePositive * leakageRate + labor
    val FPLoss = falsePositive * reOrderRate * lostOrders * juiceAdj * cancelRatio + labor
    val TNLoss = 0

    return FNLoss + TPLoss + FPLoss + TNLoss
  }
  val customLoss = CustomLoss[Float](cLoss _, sizeAverage = false)

  def lr(): KerasNet[Float] = {
    val seq = Sequential[Float]()
    seq.add(Dense[Float](1, inputShape = Shape(5)))
    seq.add(Activation[Float]("sigmoid"))
  }

  //  def testWithModel(): Unit = {
  //    val model = lr()
  //    model.compile(optimizer = SGD[Float], loss = cLoss[Float])
  //    model.fit(DataSet.array())
  //  }

  def main(args: Array[String]): Unit = {
    val yTrue = Tensor[Float](data = Array(0f, 1f, 1f, 1f, 0f), shape = Array(5, 1))
    val yPred = Tensor[Float](data = Array(0.3f, 0.6f, 0.7f, 0.2f, 0.8f), shape = Array(5, 1))
    val l = customLoss.forward(yPred, yTrue) // NOTE: the order is yPred then yTrue in forward
    println(l)
  }
}