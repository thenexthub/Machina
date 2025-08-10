###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Friday, April 11, 2025.                                             #
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
"""Experimental API for manually injecting delays into `tf.data` pipelines."""
from machina.python.data.ops import dataset_ops
from machina.python.ops import gen_experimental_dataset_ops


class _SleepDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that sleeps before producing each upstream element."""

  def __init__(self, input_dataset, sleep_microseconds):
    self._input_dataset = input_dataset
    self._sleep_microseconds = sleep_microseconds
    variant_tensor = gen_experimental_dataset_ops.sleep_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._sleep_microseconds,
        **self._flat_structure)
    super(_SleepDataset, self).__init__(input_dataset, variant_tensor)


def sleep(sleep_microseconds):
  """Sleeps for `sleep_microseconds` before producing each input element.

  Args:
    sleep_microseconds: The number of microseconds to sleep before producing an
      input element.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _SleepDataset(dataset, sleep_microseconds)

  return _apply_fn
