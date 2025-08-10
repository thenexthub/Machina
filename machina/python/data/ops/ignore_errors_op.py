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
"""The implementation of `tf.data.Dataset.ignore_errors`."""

from machina.python.data.ops import dataset_ops
from machina.python.ops import gen_experimental_dataset_ops


def _ignore_errors(input_dataset, log_warning=False, name=None):
  return _IgnoreErrorsDataset(input_dataset, log_warning, name)


class _IgnoreErrorsDataset(dataset_ops.UnaryUnchangedStructureDataset):
  """A `Dataset` that drops erroneous elements from its input."""

  def __init__(self, input_dataset, log_warning, name=None):
    """See `Dataset.ignore_errors` for details."""
    self._input_dataset = input_dataset
    self._name = name
    variant_tensor = (
        gen_experimental_dataset_ops.ignore_errors_dataset(
            self._input_dataset._variant_tensor,  # pylint: disable=protected-access
            log_warning=log_warning,
            **self._flat_structure))
    super().__init__(input_dataset, variant_tensor)
