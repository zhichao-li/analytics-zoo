package com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader

import com.fasterxml.jackson.databind.JsonNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.models.KerasNet

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


trait Loader {
  def fromConfig(config: JsonNode): AbstractModule[Activity, Activity, Float] = {
    throw new RuntimeException("Haven't been implemented yet")
  }

  def getClazz(className: String): Class[LayerLoader] = {
    Class.forName(
      s"com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader.layers.${className}Loader")
      .asInstanceOf[Class[LayerLoader]]
  }

  final def load(layerLevelConfig: JsonNode): KerasLayer[Activity, Activity, Float] = {
    val className = layerLevelConfig.get("class_name").asText()
    val cls = getClazz(className)
    val method = KerasUtils.findMethod(cls,
      "fromConfig",
      layerLevelConfig)
    val module = method.invoke(cls, layerLevelConfig).asInstanceOf[KerasLayer[Activity, Activity, Float]]
    module
  }
}

trait LayerLoader extends Loader {
  def toZooFormat(kerasWeights: Array[Tensor[Float]]): Array[Tensor[Float]]
}

object ModelLoader extends Loader {
  override def getClazz(className: String): Class[LayerLoader] = {
    Class.forName(
      s"com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader.model.${className}Loader")
      .asInstanceOf[Class[LayerLoader]]
  }

  def load(h5Path: String, resetWeights: Boolean = true): KerasNet[Float] = {
    val hDF5Reader = HDF5Reader(h5Path)
    val modelConfig = hDF5Reader.readAttribute("model_config")
    val config = Utils.toJson(modelConfig)
    val model = load(config).asInstanceOf[KerasNet[Float]]
    loadWeightsFromH5(hDF5Reader, model)
    model
  }

  def processWeightString(str: String): Array[String] = {
    val prefixes = str.split(":\\d+").filter(!_.isEmpty)
    val weightSurfix = prefixes.map {prefix =>
      (prefix + ":\\d+").r.findFirstIn(str).get
    }
    weightSurfix
  }

  // layers may containing
  def loadWeightsFromH5(hDF5Reader: HDF5Reader,
      model: KerasNet[Float]): Unit = {
    val layers: List[AbstractModule[Activity, Activity, Float]] = model.layers()
    val layersWithWeight = layers.filter(_.parameters() != null)

    val nameToLayer = layersWithWeight.map(l => (l.getName() -> l)).toMap

    val nameToKWeights = mutable.Map[String,
      ArrayBuffer[Tensor[Float]]]()
    val layerpaths = layersWithWeight.foreach { layer =>
      val layerDirPath = s"/model_weights/${layer.getName()}"
      val layerAttributePath = s"${layerDirPath}/weight_names"
      val weightPaths = processWeightString(hDF5Reader.readAttribute(layerAttributePath))
      weightPaths.foreach { path =>
        val kweight = hDF5Reader.readDataSet(s"${layerDirPath}/${path}")
        val layerName = path.split("/")(0)
        if (!nameToKWeights.contains(layerName)) {
          nameToKWeights += (layerName -> ArrayBuffer[Tensor[Float]]())
        }
        val buffer = nameToKWeights(layerName)
        buffer.append(kweight)
      }
      nameToKWeights.foreach { case (layerName, kweights) =>
        // convert keras weight to zoo
        val zlayer = nameToLayer(layerName)
        val zWeights = LayerLoader.toZooFormat(zlayer.getClass.getSimpleName, kweights.toArray)
        if (zlayer.isInstanceOf[KerasNet[Float]]) {
          loadWeightsFromH5(hDF5Reader, zlayer.asInstanceOf[KerasNet[Float]])
        } else {
          zlayer.setWeightsBias(zWeights)
        }
      }

    }

  }
}

object LayerLoader extends Loader {

  def toZooFormat(className: String, kerasWeights: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    val cls = getClazz(className)
    val method = KerasUtils.findMethod(cls,
      "toZooFormat",
      kerasWeights)
    method.invoke(cls,
      kerasWeights).asInstanceOf[Array[Tensor[Float]]]
  }
}
