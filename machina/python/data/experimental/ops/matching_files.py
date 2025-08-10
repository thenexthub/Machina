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
"""Experimental API for matching input filenames."""

from machina.python.data.ops import dataset_ops
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.framework import tensor_spec
from machina.python.ops import gen_experimental_dataset_ops as ged_ops


class MatchingFilesDataset(dataset_ops.DatasetSource):
  """A `Dataset` that list the files according to the input patterns."""

  def __init__(self, patterns):
    self._patterns = ops.convert_to_tensor(
        patterns, dtype=dtypes.string, name="patterns")
    variant_tensor = ged_ops.matching_files_dataset(self._patterns)
    super(MatchingFilesDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    return tensor_spec.TensorSpec([], dtypes.string)
