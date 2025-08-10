###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, July 14, 2025.                                              #
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
"""Tests for machina.python.framework.constant_op."""

from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format

from machina.core.framework import graph_pb2
from machina.python.eager import def_function
from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.framework import importer
from machina.python.framework import ops
from machina.python.ops import gradients_impl
from machina.python.ops.parallel_for import control_flow_ops
from machina.python.platform import test


class ConstantOpTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      dtypes.bfloat16,
      dtypes.complex128,
      dtypes.complex64,
      dtypes.double,
      dtypes.float16,
      dtypes.float32,
      dtypes.float64,
      dtypes.half,
      dtypes.int16,
      dtypes.int32,
      dtypes.int64,
      dtypes.int8,
      dtypes.qint16,
      dtypes.qint32,
      dtypes.qint8,
      dtypes.quint16,
      dtypes.quint8,
      dtypes.uint16,
      dtypes.uint32,
      dtypes.uint64,
      dtypes.uint8,
  )
  def test_convert_string_to_number(self, dtype):
    with self.assertRaises(TypeError):
      constant_op.constant("hello", dtype)

  def _make_graph_def(self, text):
    ret = graph_pb2.GraphDef()
    text_format.Parse(text, ret)
    return ret

  def test_eager_const_xla(self):

    @def_function.function(jit_compile=True)
    def f_using_eagerconst(x):
      graph_def = self._make_graph_def("""
         node { name: 'x' op: 'Const'
           attr { key: 'dtype' value { type: DT_FLOAT } }
           attr { key: 'value' value { tensor {
             dtype: DT_FLOAT tensor_shape {} float_val: NaN } } } }
         node { name: 'const' op: '_EagerConst' input: 'x:0'
                attr { key: 'T' value { type: DT_FLOAT } }}""")
      x_id = importer.import_graph_def(
          graph_def,
          input_map={"x:0": x},
          return_elements=["const"],
          name="import")[0].outputs[0]
      return x_id

    self.assertAllClose(3.14, f_using_eagerconst(constant_op.constant(3.14)))

  def test_np_array_memory_not_shared(self):
    # An arbitrarily large loop number to test memory sharing
    for _ in range(10000):
      x = np.arange(10)
      xt = constant_op.constant(x)
      x[3] = 42
      # Changing the input array after `xt` is created should not affect `xt`
      self.assertEqual(xt.numpy()[3], 3)

  def test_eager_const_grad_error(self):

    @def_function.function
    def f_using_eagerconst():
      x = constant_op.constant(1.)
      graph_def = self._make_graph_def("""
         node { name: 'x' op: 'Placeholder'
                attr { key: 'dtype' value { type: DT_FLOAT } }}
         node { name: 'const' op: '_EagerConst' input: 'x:0'
                attr { key: 'T' value { type: DT_FLOAT } }}""")
      x_id = importer.import_graph_def(
          graph_def,
          input_map={"x:0": x},
          return_elements=["const"],
          name="import")[0].outputs[0]
      gradients_impl.gradients(x_id, x)
      return x_id

    with self.assertRaisesRegex(AssertionError, "Please file a bug"):
      f_using_eagerconst()

  def test_eager_const_pfor(self):

    @def_function.function
    def f_using_eagerconst():

      def vec_fn(x):
        graph_def = self._make_graph_def("""
           node { name: 'x' op: 'Const'
             attr { key: 'dtype' value { type: DT_FLOAT } }
             attr { key: 'value' value { tensor {
               dtype: DT_FLOAT tensor_shape {} float_val: 3.14 } } } }
           node { name: 'const' op: '_EagerConst' input: 'x:0'
                  attr { key: 'T' value { type: DT_FLOAT } }}""")
        return importer.import_graph_def(
            graph_def,
            input_map={"x:0": x},
            return_elements=["const"],
            name="import")[0].outputs[0]

      return control_flow_ops.vectorized_map(
          vec_fn, constant_op.constant([1., 2.]), fallback_to_while_loop=False)

    self.assertAllClose([1., 2.], f_using_eagerconst())


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
