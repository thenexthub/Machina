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
###############################################################################
"""Tests for MirroredStrategy."""

from machina.python.distribute import combinations
from machina.python.distribute import mirrored_strategy
from machina.python.distribute import multi_worker_test_base
from machina.python.distribute import strategy_test_lib
from machina.python.eager import context
from machina.python.eager import test


def get_gpus():
  gpus = context.context().list_logical_devices("GPU")
  actual_gpus = []
  for gpu in gpus:
    if "job" in gpu.name:
      actual_gpus.append(gpu.name)
  return actual_gpus


@combinations.generate(
    combinations.combine(
        distribution=[
            combinations.NamedDistribution(
                "Mirrored",
                # pylint: disable=g-long-lambda
                lambda: mirrored_strategy.MirroredStrategy(get_gpus()),
                required_gpus=1)
        ],
        mode=["eager"]))
class RemoteSingleWorkerMirroredStrategyEager(
    multi_worker_test_base.SingleWorkerTestBaseEager,
    strategy_test_lib.RemoteSingleWorkerMirroredStrategyBase):

  def _get_num_gpus(self):
    return len(get_gpus())

  def testNumReplicasInSync(self, distribution):
    self._testNumReplicasInSync(distribution)

  def testMinimizeLoss(self, distribution):
    self._testMinimizeLoss(distribution)

  def testDeviceScope(self, distribution):
    self._testDeviceScope(distribution)

  def testMakeInputFnIteratorWithDataset(self, distribution):
    self._testMakeInputFnIteratorWithDataset(distribution)

  def testMakeInputFnIteratorWithCallable(self, distribution):
    self._testMakeInputFnIteratorWithCallable(distribution)


if __name__ == "__main__":
  test.main()
