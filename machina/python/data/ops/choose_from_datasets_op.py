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
"""The implementation of `tf.data.Dataset.choose_from_datasets`."""

from machina.python.data.ops import dataset_ops
from machina.python.data.ops import directed_interleave_op
from machina.python.data.util import structure
from machina.python.framework import dtypes
from machina.python.framework import tensor_spec
from machina.python.types import data as data_types


def _choose_from_datasets(  # pylint: disable=unused-private-name
    datasets, choice_dataset, stop_on_empty_dataset=True
):
  """See `Dataset.choose_from_datasets()` for details."""

  if not datasets:
    raise ValueError("Invalid `datasets`. `datasets` should not be empty.")
  if not isinstance(choice_dataset, data_types.DatasetV2):
    raise TypeError(
        "Invalid `choice_dataset`. `choice_dataset` should be a "
        f"`tf.data.Dataset` but is {type(choice_dataset)}."
    )
  if not structure.are_compatible(
      choice_dataset.element_spec, tensor_spec.TensorSpec([], dtypes.int64)
  ):
    raise TypeError(
        "Invalid `choice_dataset`. Elements of `choice_dataset` "
        "must be scalar `tf.int64` tensors but are "
        f"{choice_dataset.element_spec}."
    )
  # Replicates the `choice_dataset` component so that each split makes choices
  # independently. This avoids the need for prohibitively expensive
  # cross-split coordination.
  # pylint: disable=protected-access
  choice_dataset = dataset_ops._apply_rewrite(
      choice_dataset, "replicate_on_split"
  )
  return directed_interleave_op._directed_interleave(  # pylint: disable=protected-access
      choice_dataset, datasets, stop_on_empty_dataset
  )
