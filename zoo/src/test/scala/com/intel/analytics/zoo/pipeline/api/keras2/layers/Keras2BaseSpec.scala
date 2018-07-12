package com.intel.analytics.zoo.pipeline.api.keras2.layers

import com.fasterxml.jackson.databind.JsonNode
import com.intel.analytics.zoo.pipeline.api.keras.layers.KerasBaseSpec
import com.intel.analytics.zoo.pipeline.api.keras2.layers.model.loader.{HDF5Reader, LayerLoader, Utils}

class Keras2ModelLoadingBaseSpec extends KerasBaseSpec{

  private[zoo] def randomModelPath() = {
    val tmpFile = createTmpFile()
    val modelPath = s"${tmpFile.getAbsolutePath}.h5"
    println(s"modelPath: ${modelPath}")
    modelPath
  }
  // Retrieve the config for the first layer
  private[zoo] def getLayerConfigForSeq(modelPath: String, index: Int) = {
    val modelConfig = new HDF5Reader(modelPath).readAttribute("model_config")
    val config = Utils.toJson(modelConfig)
    config.get("config").get(index)
  }

  private[zoo] def reloadLayerForSeq(modelPath: String, index: Int) = {
    val config = getLayerConfigForSeq(modelPath, index)
    LayerLoader.load(config)
  }

  // We are suppose there's only one single output.
  private[zoo] def reloadOutputLayerForModel(modelPath: String) = {
    val h5Reader = new HDF5Reader(modelPath)
    val modelConfig = h5Reader.readAttribute("model_config")
    val config = Utils.toJson(modelConfig)
    val layers = config.get("config").get("layers").iterator()
    var outLayer: JsonNode = null
    while(layers.hasNext) {
      outLayer = layers.next()
    }
    LayerLoader.load(outLayer)
  }
}
