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
"""The implementation of `tf.data.Dataset.take_while`."""

from machina.python.data.ops import dataset_ops
from machina.python.data.ops import structured_function
from machina.python.framework import dtypes
from machina.python.framework import tensor_spec
from machina.python.ops import gen_experimental_dataset_ops as ged_ops


def _take_while(input_dataset, predicate, name=None):  # pylint: disable=unused-private-name
  """See `Dataset.take_while()` for details."""
  return _TakeWhileDataset(input_dataset, predicate, name=name)


class _TakeWhileDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A dataset that stops iteration when `predicate` returns false."""

  def __init__(self, input_dataset, predicate, name=None):
    """See `take_while()` for details."""

    self._input_dataset = input_dataset
    wrapped_func = structured_function.StructuredFunctionWrapper(
        predicate, self._transformation_name(), dataset=self._input_dataset)

    if not wrapped_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.bool)):
      raise ValueError(f"Invalid `predicate`. `predicate` must return a "
                       f"`tf.bool` scalar tensor but its return type is"
                       f"{wrapped_func.output_structure}.")

    self._predicate = wrapped_func
    self._name = name
    variant_tensor = ged_ops.take_while_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        other_arguments=self._predicate.function.captured_inputs,
        predicate=self._predicate.function,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._predicate]

  def _transformation_name(self):
    return "Dataset.take_while()"
