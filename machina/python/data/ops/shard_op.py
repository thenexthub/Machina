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
"""The implementation of `tf.data.Dataset.shard`."""

from machina.python.data.ops import dataset_ops
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.ops import gen_dataset_ops


def _shard(input_dataset, num_shards, index, name):  # pylint: disable=unused-private-name
  """See `Dataset.shard()` for details."""
  return _ShardDataset(input_dataset, num_shards, index, name)


class _ShardDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` for sharding its input."""

  def __init__(self, input_dataset, num_shards, index, name):
    """See `Dataset.shard()` for details."""
    self._input_dataset = input_dataset
    self._num_shards = ops.convert_to_tensor(
        num_shards, dtype=dtypes.int64, name="num_shards")
    self._index = ops.convert_to_tensor(index, dtype=dtypes.int64, name="index")
    self._name = name
    variant_tensor = gen_dataset_ops.shard_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        num_shards=self._num_shards,
        index=self._index,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)
