###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Friday, April 11, 2025.                                             #
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
"""Tests for composite_tensor_ops."""

from absl.testing import parameterized

from machina.python.eager import backprop
from machina.python.eager import context
from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.framework import errors
from machina.python.framework import sparse_tensor
from machina.python.framework import test_util
from machina.python.ops import composite_tensor_ops
from machina.python.ops import gen_composite_tensor_ops
from machina.python.ops import gen_list_ops
from machina.python.ops import gradients_impl
from machina.python.ops import math_ops
from machina.python.ops import parsing_ops
from machina.python.ops import sparse_ops
from machina.python.ops.ragged import ragged_factory_ops
from machina.python.ops.ragged import ragged_tensor
from machina.python.platform import googletest
from machina.python.util import nest


@test_util.run_all_in_graph_and_eager_modes
class ExtensionTypeTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      ('Ragged', lambda: ragged_factory_ops.constant([[1, 2], [3], [4, 5, 6]])),
      ('Sparse', lambda: sparse_ops.from_dense([[0, 0, 3, 0], [1, 2, 0, 0]])),
  ])
  def testEncodeAndDecode(self, value_factory):
    value = value_factory()

    encoded = composite_tensor_ops.composite_tensor_to_variants(value)
    self.assertEqual(encoded.dtype, dtypes.variant)
    self.assertEqual(encoded.shape.rank, 0)

    decoded = composite_tensor_ops.composite_tensor_from_variant(
        encoded, value._type_spec)
    self.assertTrue(value._type_spec.is_compatible_with(decoded._type_spec))
    value_components = nest.flatten(value, expand_composites=True)
    decoded_components = nest.flatten(decoded, expand_composites=True)
    self.assertLen(value_components, len(decoded_components))
    for v, d in zip(value_components, decoded_components):
      self.assertAllEqual(v, d)

  @parameterized.named_parameters([
      ('WrongType', lambda: ragged_factory_ops.constant([[1]]),
       sparse_tensor.SparseTensorSpec([None, None], dtypes.int32),
       r'Expected a SPARSE_TENSOR_SPEC \(based on `type_spec`\), but `encoded` '
       'contains a RAGGED_TENSOR_SPEC'),
      ('WrongNumComponents', lambda: ragged_factory_ops.constant([[1]]),
       ragged_tensor.RaggedTensorSpec([None, None, None], dtypes.int32),
       'Encoded value has 2 tensor components; expected 3 components'),
      ('WrongDType', lambda: ragged_factory_ops.constant([[1]]),
       ragged_tensor.RaggedTensorSpec([None, None], dtypes.float32),
       'Tensor component 0 had dtype DT_INT32; expected dtype DT_FLOAT'),
  ])
  def testDecodingErrors(self, value, spec, message):
    encoded = composite_tensor_ops.composite_tensor_to_variants(value())
    with self.assertRaisesRegex(errors.InvalidArgumentError, message):
      self.evaluate(
          composite_tensor_ops.composite_tensor_from_variant(encoded, spec))

  @parameterized.named_parameters([
      ('IncompatibleSpec', lambda: ragged_factory_ops.constant([[1]]),
       ragged_tensor.RaggedTensorSpec([None, None, None], dtypes.int32),
       r'`type_spec` .* is not compatible with `value` .*'),
  ])
  def testEncodingErrors(self, value, spec, message):
    with self.assertRaisesRegex(ValueError, message):
      composite_tensor_ops.composite_tensor_to_variants(value(), spec)

  def testDecodingEmptyNonScalarTensorError(self):
    if not context.executing_eagerly():
      # Creating a variant tensor of an empty list is not allowed in eager mode.
      return

    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'must not be an empty variant tensor'):
      gen_composite_tensor_ops.CompositeTensorVariantToComponents(
          encoded=constant_op.constant([], dtype=dtypes.variant),
          metadata='',
          Tcomponents=[dtypes.int32])

  def testDecodingInvalidEncodedInputError(self):
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'not a valid CompositeTensorVariant tensor'):
      self.evaluate(
          gen_composite_tensor_ops.CompositeTensorVariantToComponents(
              encoded=gen_list_ops.EmptyTensorList(
                  element_dtype=dtypes.int32,
                  element_shape=[1, 2],
                  max_num_elements=2),
              metadata='',
              Tcomponents=[dtypes.int32]))

  def testRoundTripThroughTensorProto(self):
    value = ragged_factory_ops.constant([[1, 2], [3], [4, 5, 6]])
    encoded = composite_tensor_ops.composite_tensor_to_variants(value)
    proto = parsing_ops.SerializeTensor(tensor=encoded)
    parsed = parsing_ops.ParseTensor(serialized=proto, out_type=dtypes.variant)
    decoded = composite_tensor_ops.composite_tensor_from_variant(
        parsed, value._type_spec)
    self.assertAllEqual(value, decoded)

  def testGradient(self):

    def func(x):
      x2 = composite_tensor_ops.composite_tensor_to_variants(x * 2)
      x3 = composite_tensor_ops.composite_tensor_from_variant(x2, x._type_spec)
      return x3.with_values(x3.values * math_ops.range(6.0))

    x = ragged_factory_ops.constant([[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]])
    if context.executing_eagerly():
      with backprop.GradientTape() as t:
        t.watch(x.values)
        y = func(x)
        g = t.gradient(y.values, x.values)
    else:
      y = func(x)
      g = gradients_impl.gradients(ys=y.values, xs=x.values)[0]
    self.assertAllClose(g, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])


if __name__ == '__main__':
  googletest.main()
