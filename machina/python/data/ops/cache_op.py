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
"""The implementation of `tf.data.Dataset.cache`."""

from machina.python import tf2
from machina.python.data.ops import dataset_ops
from machina.python.eager import context
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.ops import gen_dataset_ops


def _cache(input_dataset, filename, name):  # pylint: disable=unused-private-name
  return CacheDataset(input_dataset, filename, name)


class CacheDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that caches elements of its input."""

  def __init__(self, input_dataset, filename, name=None):
    """See `Dataset.cache()` for details."""
    self._input_dataset = input_dataset
    self._filename = ops.convert_to_tensor(
        filename, dtype=dtypes.string, name="filename")
    self._name = name
    if tf2.enabled() and (context.executing_eagerly() or ops.inside_function()):
      variant_tensor = gen_dataset_ops.cache_dataset_v2(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          filename=self._filename,
          cache=gen_dataset_ops.dummy_memory_cache(),
          **self._common_args)
    else:
      variant_tensor = gen_dataset_ops.cache_dataset(
          input_dataset._variant_tensor,  # pylint: disable=protected-access
          filename=self._filename,
          **self._common_args)
    super().__init__(input_dataset, variant_tensor)
