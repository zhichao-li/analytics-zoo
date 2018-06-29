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

package com.intel.analytics.zoo.pipeline.api.keras.layers.utils

import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.keras.{KerasIdentityWrapper, KerasLayer, KerasLayerWrapper, Sequential => KSequential, SoftMax => KSoftMax}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.metrics.AUC
import com.intel.analytics.zoo.pipeline.api.keras.models.KerasNet

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object KerasUtils {

  // stack on the first dimension
  def stackTensor[T: ClassTag](
      tensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val totalSize = tensors.map(_.size().product).sum
    val mergedStorage = new Array[T](totalSize)
    var mergedOffSet = 0
    tensors.map {tensor =>
      require(tensor.isContiguous(), "The tensor should be contiguous")
      require(tensor.size().drop(1).deep == tensors.head.size().drop(1).deep,
        s"The shape of tensor should be the same," +
          s"but got: ${tensor.size().mkString(",")} " +
          s"and ${tensors.head.size().drop(1).mkString(",")}")
      val offset = tensor.storageOffset() - 1
      System.arraycopy(tensor.storage().array(), offset, mergedStorage, mergedOffSet,
        tensor.size().product)
      mergedOffSet += tensor.size().product
    }
    println(mergedStorage(3))
    Tensor[T](shape = Array(tensors.map(_.size(1)).sum) ++ tensors.head.size().drop(1),
      data = mergedStorage)
  }


  /**
   * A keras-like API for local prediction.
   * The first dim of x should be the number of samples.
   * @param x if the model accept 2 inputs then the length of the array should be 2.
   * @param batch the batch size when executing the inference.
   */
  def predict[T: ClassTag](module: AbstractModule[Activity, Activity, T],
      x: Array[Tensor[T]], batch: Int)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val preds = KerasUtils.predictMultiOut[T](module, x, batch)
    println("yuiyui")
    KerasUtils.stackTensor[T](preds.map(_.toTensor))
  }

  def predictMultiOut[T: ClassTag](module: AbstractModule[Activity, Activity, T],
      x: Array[Tensor[T]], batch: Int)(implicit ev: TensorNumeric[T]): Array[Activity] = {
    val splitedFeatures = x.map(_.split(size = batch, dim = 1))
    var j = 0
    val inputs = ArrayBuffer[Array[Tensor[T]]]()
    while(j < splitedFeatures(0).length) {
      inputs.append(Range(0, x.length).map(recordId => splitedFeatures(recordId)(j)).toArray)
      j += 1
    }
    val modelInputs = inputs.map { features =>
      if (features.length == 1) {
        features(0).asInstanceOf[Activity]
      } else {
        T.array(features).asInstanceOf[Activity]
      }
    }
    val preds = modelInputs.map { input =>
      val pred = module.forward(input)
      if (pred.isInstanceOf[Table]) {
        T.array(pred.toTable.toTable.toSeq[Tensor[T]].map(_.clone()).toArray)
      } else if (pred.isInstanceOf[Tensor[T]]) {
        pred.toTensor.clone().asInstanceOf[Activity]
      } else {
        throw new IllegalArgumentException(s"Not supported type: ${pred.getClass}")
      }
    }
    preds.toArray
  }

  def getPadsFromBorderMode(borderMode: String = "valid"): (Int, Int) = {
    if (borderMode == "same") {
      // padH, padW
      (-1, -1)
    } else {
      (0, 0)
    }
  }

  def getInitMethod(init: String): InitializationMethod = {
    init.toLowerCase() match {
      case "glorot_uniform" => Xavier
      case "one" => Ones
      case "zero" => Zeros
      case "uniform" => RandomUniform(-0.05, 0.05)
      case "normal" => RandomNormal(0.0, 0.05)
      case _ => throw new IllegalArgumentException(s"Unsupported initialization method: " +
        s"${init.toLowerCase()}")
    }
  }

  def getKerasActivation[T : ClassTag] (activation: String)
    (implicit ev: TensorNumeric[T]): KerasLayer[Tensor[T], Tensor[T], T] = {
    if (activation == null) { return null }
    if (activation.toLowerCase() == "softmax") {
      KSoftMax[T]()
    } else {
      val torchActivation = getTorchActivation(activation)
      new KerasIdentityWrapper[T](torchActivation)
        .asInstanceOf[KerasLayer[Tensor[T], Tensor[T], T]]
    }
  }

  def getTorchActivation[T : ClassTag] (activation: String)
    (implicit ev: TensorNumeric[T]): AbstractModule[Tensor[T], Tensor[T], T] = {
    if (activation == null) null
    else {
      activation.toLowerCase() match {
          case "tanh" => Tanh[T]()
          case "sigmoid" => Sigmoid[T]()
          case "relu" => ReLU[T]()
          case "softmax" =>
                com.intel.analytics.bigdl.nn.SoftMax[T]()
          case "softplus" => SoftPlus[T]()
          case "softsign" => SoftSign[T]()
          case "hard_sigmoid" => HardSigmoid[T]()
          case "relu6" => ReLU6[T]()
          case "tanh_shrink" => TanhShrink[T]()
          case "softmin" => SoftMin[T]()
          case "log_sigmoid" => LogSigmoid[T]()
          case "log_softmax" => LogSoftMax[T]()
          case "linear" => Identity[T]().asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
          case _ => throw new IllegalArgumentException(s"Invalid activation: " +
            s"${activation.toLowerCase}. Only simple activations can be constructed using string")
      }
    }
  }

  def computeConvOutputLength(
    inputLength: Int,
    filterSize: Int,
    borderMode: String,
    stride: Int,
    dilation: Int = 1): Int = {
    val dilatedFilterSize = filterSize + (filterSize - 1) * (dilation - 1)
    val outputLength = borderMode match {
      case "valid" => inputLength - dilatedFilterSize + 1
      case "same" => inputLength
    }
    (outputLength + stride - 1) / stride
  }

  def getPadsFromBorderMode3D(
    borderMode: String = "valid"): (Int, Int, Int) = {
    if (borderMode == "same") {
      // padT, padH, padW
      (-1, -1, -1)
    } else {
      (0, 0, 0)
    }
  }

  def toBigDLFormat(dimOrdering: String): DataFormat = {
    require(dimOrdering.toLowerCase() == "tf" || dimOrdering.toLowerCase() == "th",
      s"Dim ordering must be either tf or th, but got ${dimOrdering.toLowerCase()}")
    dimOrdering.toLowerCase() match {
      case "tf" => DataFormat.NHWC
      case "th" => DataFormat.NCHW
    }
  }

  def toBigDLFormat5D(dimOrdering: String): String = {
    require(dimOrdering.toLowerCase() == "tf" || dimOrdering.toLowerCase() == "th",
      s"Dim ordering must be either tf or th, but got ${dimOrdering.toLowerCase()}")
    dimOrdering.toLowerCase() match {
      case "tf" => "CHANNEL_LAST"
      case "th" => "CHANNEL_FIRST"
    }
  }

  def toBigDLCriterion[T : ClassTag](loss: String)
    (implicit ev: TensorNumeric[T]): Criterion[T] = {
    loss.toLowerCase() match {
      case "binary_crossentropy" => BCECriterion[T]()
      case "categorical_crossentropy" => CategoricalCrossEntropy[T]()
      case "mse" => MSECriterion[T]()
      case "mean_squared_error" => MSECriterion[T]()
      case "mae" => AbsCriterion[T]()
      case "mean_absolute_error" => AbsCriterion[T]()
      case "hinge" => MarginCriterion[T]()
      case "mape" => MeanAbsolutePercentageCriterion[T]()
      case "mean_absolute_percentage_error" => MeanAbsolutePercentageCriterion[T]()
      case "msle" => MeanSquaredLogarithmicCriterion[T]()
      case "mean_squared_logarithmic_error" => MeanSquaredLogarithmicCriterion[T]()
      case "squared_hinge" => MarginCriterion[T](squared = true)
      case "sparse_categorical_crossentropy" => ClassNLLCriterion[T](logProbAsInput = false)
      case "kld" => KullbackLeiblerDivergenceCriterion[T]()
      case "kullback_leibler_divergence" => KullbackLeiblerDivergenceCriterion[T]()
      case "cosine_proximity" => CosineProximityCriterion[T]()
      case "poisson" => PoissonCriterion[T]()
      case _ => throw new IllegalArgumentException(s"Invalid loss: ${loss.toLowerCase()}")
    }
  }

  def toBigDLOptimMethod[T: ClassTag](optimMethod: String)
    (implicit ev: TensorNumeric[T]): OptimMethod[T] = {
    optimMethod.toLowerCase() match {
      case "sgd" => new SGD[T](learningRate = 0.01)
      case "rmsprop" => new RMSprop[T](learningRate = 0.001, decayRate = 0.9)
      case "adamax" => new Adamax[T](Epsilon = 1e-8)
      case "adagrad" => new Adagrad[T](learningRate = 0.01)
      case "adadelta" => new Adadelta[T](decayRate = 0.95, Epsilon = 1e-8)
      case "adam" => new Adam[T]()
    }
  }

  def toBigDLMetrics[T: ClassTag](metrics: List[String])
    (implicit ev: TensorNumeric[T]): List[ValidationMethod[T]] = {
    if (metrics == null) {
      null
    } else {
      metrics.map { metric =>
        metric.toLowerCase() match {
          case "accuracy" => new Top1Accuracy[T]()
          case "mae" => new MAE[T]()
          case "auc" => new AUC[T]()
          case "loss" => new Loss[T]()
          case "treennaccuracy" => new TreeNNAccuracy[T]()
          case _ => throw new IllegalArgumentException(s"Unsupported metric: ${metric}")
        }
      }
    }
  }

  def addBatch(shape: Shape): Shape = {
    if (shape == null) {
      return null
    }
    if (shape.isInstanceOf[SingleShape]) {
      Shape((List(-1) ++ shape.toSingle()).toArray)
    } else {
      Shape(shape.toMulti().map {addBatch})
    }
  }

  def removeBatch(shape: Shape): Shape = {
    if (shape == null) {
      return null
    }
    if (shape.isInstanceOf[SingleShape]) {
      Shape(shape.toSingle().slice(1, shape.toSingle().length).toArray)
    } else {
      Shape(shape.toMulti().map {removeBatch})
    }
  }

  def fuse[T: ClassTag](
      torchLayer: AbstractModule[Activity, Activity, T],
      kerasActivation: KerasLayer[Tensor[T], Tensor[T], T],
      batchInputShape: Shape)
      (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    if (kerasActivation == null) {
      torchLayer
    } else {
      val wrapper = KSequential[T]()
      wrapper.add(new KerasLayerWrapper[T](torchLayer,
        removeBatch(batchInputShape)))
      wrapper.add(kerasActivation)
      wrapper.setName(torchLayer.getName())
      wrapper.build(batchInputShape)
      wrapper
    }
  }

  private[zoo] def invokeMethod(obj: Object, methodName: String, args: Object*): Object = {
    val clazz = obj.getClass()
    val method =
      try {
      clazz.getMethod(methodName, args.map(_.getClass): _*)
    } catch {
        case t: Throwable =>
          val methods = clazz.getMethods().filter(_.getName() == methodName)
          require(methods.length == 1,
            s"We should only found one result, but got ${methodName}: ${methods.length}")
          methods(0)
    }
    method.invoke(obj, args: _*)
  }

  /**
   * Count the total number of parameters for a KerasLayer.
   * Return a tuple (total params #, trainable params #)
   */
  def countParams[T: ClassTag](layer: KerasLayer[Activity, Activity, T]): (Int, Int) = {
    val (weights, gradWeights) = layer.parameters()
    var count = 0
    for (w <- weights) {
      count += w.nElement()
    }
    if (layer.isInstanceOf[KerasNet[T]]) {
      val modules = layer.labor.asInstanceOf[Container[Activity, Activity, T]].modules
      var trainable = 0
      for (module <- modules) {
        trainable += countParams[T](module.asInstanceOf[KerasLayer[Activity, Activity, T]])._2
      }
      (count, trainable)
    }
    else {
      if (layer.asInstanceOf[Net].isFrozen()) {
        (count, 0)
      }
      else {
        (count, count)
      }
    }
  }

  /**
   * Return the layer summary information as an array of String, in the order of:
   * Layer (type), OutputShape, Param #
   */
  def getLayerSummary[T: ClassTag](layer: KerasLayer[Activity, Activity, T]): Array[String] = {
    val outputShape = strShape(layer.getOutputShape())
    val name = layer.getName
    val className = layer.getClass.getSimpleName
    Array(name + " (" + className + ")", outputShape.toString,
      KerasUtils.countParams(layer)._1.toString)
  }

  /**
   * Together with the layer summary of a node, also return the name of the node(s)
   * that it is connected to.
   * If there are multiple connected nodes, they will be combined by ", "
   */
  def getNodeSummary[T: ClassTag](node: ModuleNode[T]): Array[String] = {
    val layer = node.element.asInstanceOf[KerasLayer[Activity, Activity, T]]
    val results = getLayerSummary(layer)
    var connectedTo = ""
    val prevNodes = node.prevNodes
    for (i <- prevNodes.indices) {
      if (i > 0) connectedTo += ", "
      connectedTo += prevNodes(i).element.getName
    }
    results ++ Array(connectedTo)
  }

  /**
   * Print the summary of a node in a line.
   * Return a tuple (total params #, trainable params #) of this node.
   */
  def printNodeSummary[T: ClassTag](
      node: ModuleNode[T],
      lineLength: Int = 120,
      positions: Array[Double] = Array(.33, .55, .67, 1)): (Int, Int) = {
    printRow(getNodeSummary(node), lineLength, positions)
    countParams(node.element.asInstanceOf[KerasLayer[Activity, Activity, T]])
  }

  /**
   * Print a row containing several fields.
   *
   * @param fields The fields to be printed out.
   * @param lineLength The total length of a printed line.
   * @param positions The maximum absolute length proportion(%) of each field.
   *                  Default is Array(.33, .55, .67, 1), meaning that
   *                  the first field will occupy up to 33% of lineLength,
   *                  the second field will occupy up to (55-33)% of lineLength,
   *                  the third field will occupy up to (67-55)% of lineLength,
   *                  the fourth field will occupy the remaining line (100-67)%.
   *                  If the field has a larger length, the remaining part will be trimmed.
   *                  If the field has a smaller length, the remaining part will be white spaces.
   * @param includeSplitLine Whether to add a split line after printing one row.
   * @param splitChar The character to compose the split line.
   */
  def printRow(
      fields: Array[String],
      lineLength: Int = 120,
      positions: Array[Double] = Array(.33, .55, .67, 1),
      includeSplitLine: Boolean = true,
      splitChar: Char = '_'): Unit = {
    val fieldLengths = ArrayBuffer[Int]()
    for (i <- positions.indices) {
      if (i > 0) {
        val len = (positions(i) - positions(i-1)) * lineLength
        require(len > 0, s"Invalid positions specified: ${positions(i)} < ${positions(i-1)}")
        fieldLengths.append(len.toInt)
      }
      else fieldLengths.append((positions(i)*lineLength).toInt)
    }
    var line = ""
    // If there are multiple connected to nodes, each will be printed in a separate line.
    var nodes = Array[String]()
    for (i <- fields.indices) {
      if (i > 0) line += " "
      if (i == 3) {
        nodes = fields(i).split(", ")
        line += nodes(0)
      }
      else {
        line += fields(i)
      }
      val maxLength = fieldLengths.take(i + 1).sum
      if (line.length > maxLength) {
        line = line.substring(0, maxLength)
      }
      else {
        line += " " * (maxLength - line.length)
      }

    }
    println(line)
    // If there are multiple connected to nodes, print the remaining each in a separate line
    // without the split line.
    for (node <- nodes.slice(1, nodes.length)) {
      printRow(Array("", "", "", node), lineLength, positions, includeSplitLine = false)
    }
    if (includeSplitLine) printSplitLine(splitChar, lineLength)
  }

  /**
   * Print a split line that repeats the 'char' for 'lineLength' times.
   */
  def printSplitLine(char: Char, lineLength: Int = 120): Unit = {
    val str = char.toString
    println(str * lineLength)
  }

  /**
   * Convert a Shape to String format using 'None' to indicate batch,
   * which is the same as Keras. Used to print out the shape.
   *
   * For example,
   * (None, 10) will be returned for Shape(-1, 10), a SingleShape.
   * (None, 10) (None, 8) will be returned for a MultiShape which consists of
   * Shape(-1, 10), Shape(-1, 8).
   */
  def strShape(shape: Shape): String = {
    shape match {
      case s: SingleShape =>
        val res = "(" + s.toSingle().mkString(", ") + ")"
        res.replaceFirst("-1", "None")
      case m: MultiShape =>
        val shapes = m.toMulti()
        var res = ""
        for (shape <- shapes) res = res + strShape(shape) + " "
        res
    }
  }
}
