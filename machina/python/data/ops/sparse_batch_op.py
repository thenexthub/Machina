###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, May 19, 2025.                                               #
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
"""The implementation of `tf.data.Dataset.sparse_batch`."""
from machina.python.data.ops import dataset_ops
from machina.python.data.util import convert
from machina.python.framework import dtypes
from machina.python.framework import sparse_tensor
from machina.python.framework import tensor_shape
from machina.python.ops import gen_experimental_dataset_ops as ged_ops


def _sparse_batch(input_dataset, batch_size, row_shape, name=None):
  return _DenseToSparseBatchDataset(input_dataset, batch_size, row_shape, name)


class _DenseToSparseBatchDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that batches ragged dense elements into `tf.sparse.SparseTensor`s."""

  def __init__(self, input_dataset, batch_size, row_shape, name=None):
    """See `Dataset.dense_to_sparse_batch()` for more details."""
    if not isinstance(
        dataset_ops.get_legacy_output_types(input_dataset), dtypes.DType):
      raise TypeError("`dense_to_sparse_batch` requires an input dataset whose "
                      "elements have a single component, but the given dataset "
                      "has the following component types: "
                      f"{dataset_ops.get_legacy_output_types(input_dataset)}.")
    self._input_dataset = input_dataset
    self._batch_size = batch_size
    self._row_shape = row_shape
    self._element_spec = sparse_tensor.SparseTensorSpec(
        tensor_shape.TensorShape([None]).concatenate(self._row_shape),
        dataset_ops.get_legacy_output_types(input_dataset))
    self._name = name
    variant_tensor = ged_ops.dense_to_sparse_batch_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._batch_size,
        row_shape=convert.partial_shape_to_tensor(self._row_shape),
        **self._flat_structure)
    super(_DenseToSparseBatchDataset, self).__init__(input_dataset,
                                                     variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec
