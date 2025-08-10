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
"""Datasets for random number generators."""
import functools

from machina.python import tf2
from machina.python.compat import v2_compat
from machina.python.data.ops import dataset_ops
from machina.python.data.ops import random_op
from machina.python.util import deprecation
from machina.python.util.tf_export import tf_export


# TODO(b/260143413): Migrate users to `tf.data.Dataset.random`.
@deprecation.deprecated(None, "Use `tf.data.Dataset.random(...)`.")
@tf_export("data.experimental.RandomDataset", v1=[])
class RandomDatasetV2(random_op._RandomDataset):  # pylint: disable=protected-access
  """A `Dataset` of pseudorandom values."""


@deprecation.deprecated(None, "Use `tf.data.Dataset.random(...)`.")
@tf_export(v1=["data.experimental.RandomDataset"])
class RandomDatasetV1(dataset_ops.DatasetV1Adapter):
  """A `Dataset` of pseudorandom values."""

  @functools.wraps(RandomDatasetV2.__init__)
  def __init__(self, seed=None):
    wrapped = RandomDatasetV2(seed)
    super(RandomDatasetV1, self).__init__(wrapped)


if tf2.enabled():
  RandomDataset = RandomDatasetV2
else:
  RandomDataset = RandomDatasetV1


def _tf2_callback():
  global RandomDataset
  if tf2.enabled():
    RandomDataset = RandomDatasetV2
  else:
    RandomDataset = RandomDatasetV1


v2_compat.register_data_v2_callback(_tf2_callback)
