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
"""The implementation of `tf.data.Dataset.range`."""

from machina.python.data.ops import dataset_ops
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.framework import tensor_spec
from machina.python.ops import gen_dataset_ops


def _range(*args, **kwargs):  # pylint: disable=unused-private-name
  return _RangeDataset(*args, **kwargs)


class _RangeDataset(dataset_ops.DatasetSource):
  """A `Dataset` of a step separated range of values."""

  def __init__(self, *args, **kwargs):
    """See `Dataset.range()` for details."""
    self._parse_args(*args, **kwargs)
    self._structure = tensor_spec.TensorSpec([], self._output_type)
    variant_tensor = gen_dataset_ops.range_dataset(
        start=self._start,
        stop=self._stop,
        step=self._step,
        **self._common_args)
    super().__init__(variant_tensor)

  def _parse_args(self, *args, **kwargs):
    """Parses arguments according to the same rules as the `range()` builtin."""
    if len(args) == 1:
      self._start = self._build_tensor(0, "start")
      self._stop = self._build_tensor(args[0], "stop")
      self._step = self._build_tensor(1, "step")
    elif len(args) == 2:
      self._start = self._build_tensor(args[0], "start")
      self._stop = self._build_tensor(args[1], "stop")
      self._step = self._build_tensor(1, "step")
    elif len(args) == 3:
      self._start = self._build_tensor(args[0], "start")
      self._stop = self._build_tensor(args[1], "stop")
      self._step = self._build_tensor(args[2], "step")
    else:
      raise ValueError(f"Invalid `args`. The length of `args` should be "
                       f"between 1 and 3 but was {len(args)}.")
    if "output_type" in kwargs:
      self._output_type = kwargs["output_type"]
    else:
      self._output_type = dtypes.int64
    self._name = kwargs["name"] if "name" in kwargs else None

  def _build_tensor(self, int64_value, name):
    return ops.convert_to_tensor(int64_value, dtype=dtypes.int64, name=name)

  @property
  def element_spec(self):
    return self._structure
