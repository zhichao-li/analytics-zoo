package com.intel.analytics.zoo.pipeline.api.keras2.layers


import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.Merge
import com.intel.analytics.zoo.pipeline.api.keras.{layers => klayers1}

import scala.reflect.ClassTag

class Add[T: ClassTag](
    override val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends Merge[T](layers = null, mode = "add", inputShape = inputShape) with Net {

}
