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
"""The implementation of `tf.data.Dataset.flat_map`."""

from machina.python.data.ops import dataset_ops
from machina.python.data.ops import structured_function
from machina.python.ops import gen_dataset_ops


def _flat_map(input_dataset, map_func, name=None):  # pylint: disable=unused-private-name
  """See `Dataset.flat_map()` for details."""
  return _FlatMapDataset(input_dataset, map_func, name)


class _FlatMapDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that maps a function over its input and flattens the result."""

  def __init__(self, input_dataset, map_func, name=None):

    self._input_dataset = input_dataset
    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func, self._transformation_name(), dataset=input_dataset)
    if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):
      raise TypeError(
          "The `map_func` argument must return a `Dataset` object. Got "
          f"{dataset_ops.get_type(self._map_func.output_structure)!r}.")
    # pylint: disable=protected-access
    self._structure = self._map_func.output_structure._element_spec
    self._name = name
    variant_tensor = gen_dataset_ops.flat_map_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._structure

  def _transformation_name(self):
    return "Dataset.flat_map()"
