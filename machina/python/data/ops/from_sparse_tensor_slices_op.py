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
###############################################################################=
"""The implementation of `tf.data.Dataset.from_sparse_tensor_slices`."""

from machina.python.data.ops import dataset_ops
from machina.python.framework import dtypes
from machina.python.framework import sparse_tensor as sparse_tensor_lib
from machina.python.framework import tensor_spec
from machina.python.ops import gen_dataset_ops


def _from_sparse_tensor_slices(sparse_tensor):  # pylint: disable=unused-private-name
  return dataset_ops.DatasetV1Adapter(_SparseTensorSliceDataset(sparse_tensor))


class _SparseTensorSliceDataset(dataset_ops.DatasetSource):
  """A `Dataset` that splits a rank-N `tf.sparse.SparseTensor` into its rows."""

  def __init__(self, sparse_tensor):
    """See `Dataset.from_sparse_tensor_slices()` for details."""
    if not isinstance(sparse_tensor, sparse_tensor_lib.SparseTensor):
      raise TypeError(f"Invalid `sparse_tensor`. `sparse_tensor` must be a "
                      f"`tf.sparse.SparseTensor`. Got {type(sparse_tensor)}.")
    self._sparse_tensor = sparse_tensor

    indices_shape = self._sparse_tensor.indices.get_shape()
    shape_shape = self._sparse_tensor.dense_shape.get_shape()
    rank = (indices_shape.dims[1] - 1).merge_with(shape_shape.dims[0] - 1)
    self._structure = (tensor_spec.TensorSpec([None, rank], dtypes.int64),
                       tensor_spec.TensorSpec([None],
                                              self._sparse_tensor.dtype),
                       tensor_spec.TensorSpec([rank], dtypes.int64))

    variant_tensor = gen_dataset_ops.sparse_tensor_slice_dataset(
        self._sparse_tensor.indices, self._sparse_tensor.values,
        self._sparse_tensor.dense_shape)
    super().__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._structure
