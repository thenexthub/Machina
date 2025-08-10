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
"""The implementation of `tf.data.Dataset.from_tensor_slices`."""

from machina.python.data.ops import dataset_ops
from machina.python.data.util import nest
from machina.python.data.util import structure
from machina.python.framework import tensor_shape
from machina.python.ops import gen_dataset_ops


def _from_tensor_slices(tensors, name=None):
  return _TensorSliceDataset(tensors, name=name)


class _TensorSliceDataset(dataset_ops.DatasetSource):
  """A `Dataset` of slices from a dataset element."""

  def __init__(self, element, is_files=False, name=None):
    """See `Dataset.from_tensor_slices` for details."""
    element = structure.normalize_element(element)
    batched_spec = structure.type_spec_from_value(element)
    self._tensors = structure.to_batched_tensor_list(batched_spec, element)
    if not self._tensors:
      raise ValueError("Invalid `element`. `element` should not be empty.")
    self._structure = nest.map_structure(
        lambda component_spec: component_spec._unbatch(), batched_spec)  # pylint: disable=protected-access
    self._name = name

    batch_dim = tensor_shape.Dimension(
        tensor_shape.dimension_value(self._tensors[0].get_shape()[0]))
    for t in self._tensors[1:]:
      batch_dim.assert_is_compatible_with(
          tensor_shape.Dimension(
              tensor_shape.dimension_value(t.get_shape()[0])))

    variant_tensor = gen_dataset_ops.tensor_slice_dataset(
        self._tensors,
        output_shapes=structure.get_flat_tensor_shapes(self._structure),
        is_files=is_files,
        metadata=self._metadata.SerializeToString())
    super().__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._structure
