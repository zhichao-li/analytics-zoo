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

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

private[zoo]class InternalRecurrent[T: ClassTag](
    batchNormParams: BatchNormParams[T] = null,
    maskZero: Boolean = false
)(implicit ev: TensorNumeric[T]) extends Recurrent[T](batchNormParams, maskZero) {


  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    super.add(module)
    if (this.preTopology != null) {
      this.preTopology = this.preTopology.cloneModule()
      modules(0) = preTopology
      module.asInstanceOf[Cell[T]].preTopology = null
    }
    this
  }
}
