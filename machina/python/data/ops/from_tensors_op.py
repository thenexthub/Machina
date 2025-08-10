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
"""The implementation of `tf.data.Dataset.from_tensors`."""

from machina.python.data.ops import dataset_ops
from machina.python.data.util import structure
from machina.python.ops import gen_dataset_ops


def _from_tensors(tensors, name):  # pylint: disable=unused-private-name
  return _TensorDataset(tensors, name)


class _TensorDataset(dataset_ops.DatasetSource):
  """A `Dataset` with a single element."""

  def __init__(self, element, name=None):
    """See `tf.data.Dataset.from_tensors` for details."""
    element = structure.normalize_element(element)
    self._structure = structure.type_spec_from_value(element)
    self._tensors = structure.to_tensor_list(self._structure, element)
    self._name = name
    variant_tensor = gen_dataset_ops.tensor_dataset(
        self._tensors,
        output_shapes=structure.get_flat_tensor_shapes(self._structure),
        metadata=self._metadata.SerializeToString())
    super().__init__(variant_tensor)

  @property
  def element_spec(self):
    return self._structure
