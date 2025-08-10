###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Saturday, May 31, 2025.                                             #
#                                                                             #
#   Licensed under the Apache License, Version 2.0 (the "License");           #
#   you may not use this file except in compliance with the License.          #
#   You may obtain a copy of the License at:                                  #
#                                                                             #
#       http://www.apache.org/licenses/LICENSE-2.0                            #
#                                                                             #
#   Unless required by applicable law or agreed to in writing, software       #
#   distributed under the License is distributed on an "AS IS" BASIS,         #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
#   See the License for the specific language governing permissions and       #
#   limitations under the License.                                            #
#                                                                             #
#   Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,            #
#   Middletown, DE 19709, New Castle County, USA.                             #
#                                                                             #
###############################################################################
"""Concurrency tests for quantize_model."""

from concurrent import futures

import numpy as np
import machina  # pylint: disable=unused-import

from machina.compiler.mlir.quantization.machina import quantization_options_pb2 as quant_opts_pb2
from machina.compiler.mlir.quantization.machina.python import quantize_model
from machina.python.eager import def_function
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.framework import tensor_spec
from machina.python.framework import test_util
from machina.python.ops import math_ops
from machina.python.platform import test
from machina.python.saved_model import save as saved_model_save
from machina.python.saved_model import tag_constants
from machina.python.trackable import autotrackable


class MultiThreadedTest(test.TestCase):
  """Tests involving multiple threads."""

  def setUp(self):
    super(MultiThreadedTest, self).setUp()
    self.pool = futures.ThreadPoolExecutor(max_workers=4)

  def _convert_with_calibration(self):
    class ModelWithAdd(autotrackable.AutoTrackable):
      """Basic model with addition."""

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  shape=[10], dtype=dtypes.float32, name='x'
              ),
              tensor_spec.TensorSpec(
                  shape=[10], dtype=dtypes.float32, name='y'
              ),
          ]
      )
      def add(self, x, y):
        res = math_ops.add(x, y)
        return {'output': res}

    def data_gen():
      for _ in range(255):
        yield {
            'x': ops.convert_to_tensor(
                np.random.uniform(size=(10)).astype('f4')
            ),
            'y': ops.convert_to_tensor(
                np.random.uniform(size=(10)).astype('f4')
            ),
        }

    root = ModelWithAdd()

    temp_path = self.create_tempdir().full_path
    saved_model_save.save(
        root, temp_path, signatures=root.add.get_concrete_function()
    )

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=quant_opts_pb2.QuantizationMethod.PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags={tag_constants.SERVING},
        signature_keys=['serving_default'],
    )

    model = quantize_model.quantize(
        temp_path,
        quantization_options=quantization_options,
        representative_dataset=data_gen(),
    )
    return model

  @test_util.run_in_graph_and_eager_modes
  def test_multiple_conversion_jobs_with_calibration(self):
    # Ensure that multiple conversion jobs with calibration won't encounter any
    # concurrency issue.
    with self.pool:
      jobs = []
      for _ in range(10):
        jobs.append(self.pool.submit(self._convert_with_calibration))

      for job in jobs:
        self.assertIsNotNone(job.result())


if __name__ == '__main__':
  test.main()
