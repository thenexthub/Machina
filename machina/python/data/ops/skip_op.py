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
"""The implementation of `tf.data.Dataset.skip`."""
from machina.python.data.ops import dataset_ops
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.ops import gen_dataset_ops


def _skip(self, count, name=None):  # pylint: disable=unused-private-name
  return _SkipDataset(self, count, name)


class _SkipDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` skipping the first `count` elements from its input."""

  def __init__(self, input_dataset, count, name=None):
    """See `Dataset.skip()` for details."""
    self._input_dataset = input_dataset
    self._count = ops.convert_to_tensor(count, dtype=dtypes.int64, name="count")
    self._name = name
    variant_tensor = gen_dataset_ops.skip_dataset(
        input_dataset._variant_tensor,  # pylint: disable=protected-access
        count=self._count,
        **self._common_args)
    super().__init__(input_dataset, variant_tensor)
