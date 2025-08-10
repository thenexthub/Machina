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
"""The implementation of `tf.data.Dataset.prefetch`."""

from machina.python.data.ops import dataset_ops
from machina.python.data.ops import debug_mode
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.ops import gen_dataset_ops


def _prefetch(input_dataset, buffer_size, name=None):  # pylint: disable=unused-private-name
  """See `Dataset.prefetch()` for details."""
  if debug_mode.DEBUG_MODE:
    return input_dataset
  return _PrefetchDataset(input_dataset, buffer_size, name=name)


class _PrefetchDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that asynchronously prefetches its input."""

  def __init__(self, input_dataset, buffer_size, slack_period=None, name=None):
    """See `Dataset.prefetch()` for details."""
    self._input_dataset = input_dataset
    if buffer_size is None:
      buffer_size = dataset_ops.AUTOTUNE
    self._buffer_size = ops.convert_to_tensor(
        buffer_size, dtype=dtypes.int64, name="buffer_size")
    self._name = name
    # pylint: disable=protected-access
    # We colocate the prefetch dataset with its input as this collocation only
    # happens automatically in graph mode.
    with ops.colocate_with(input_dataset._variant_tensor):
      variant_tensor = gen_dataset_ops.prefetch_dataset(
          input_dataset._variant_tensor,
          buffer_size=self._buffer_size,
          slack_period=slack_period,
          legacy_autotune=(buffer_size == dataset_ops.AUTOTUNE),
          **self._common_args)
    super().__init__(input_dataset, variant_tensor)
