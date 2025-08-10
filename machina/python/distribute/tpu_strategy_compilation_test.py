###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, July 14, 2025.                                              #
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
"""Tests for TPUStrategy in regards to compiling programs."""

from machina.python.distribute import tpu_strategy as tpu_lib
from machina.python.distribute.cluster_resolver import tpu_cluster_resolver
from machina.python.eager import def_function
from machina.python.eager import remote
from machina.python.eager import test
from machina.python.framework import constant_op
from machina.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", "", "Name of TPU to connect to.")
flags.DEFINE_string("project", None, "Name of GCP project with TPU.")
flags.DEFINE_string("zone", None, "Name of GCP zone with TPU.")


def get_tpu_cluster_resolver():
  resolver = tpu_cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.zone,
      project=FLAGS.project,
  )
  return resolver


def get_tpu_strategy():
  resolver = get_tpu_cluster_resolver()
  remote.connect_to_cluster(resolver)
  tpu_cluster_resolver.initialize_tpu_system(resolver)
  strategy = tpu_lib.TPUStrategyV2(resolver)
  return strategy


# TODO(b/158494076): Merge this test back into TPUStrategy tests
# (tpu_strategy_test) once MLIR bridge is enabled by default.
class TPUStrategyCompilationTest(test.TestCase):

  def test_functions_compile_same_signature(self):
    """Tests compiling different functions with the same signature."""
    strategy = get_tpu_strategy()

    @def_function.function
    def return_one():

      def computation():
        return constant_op.constant(1)

      return strategy.run(computation)

    @def_function.function
    def return_two():

      def computation():
        return constant_op.constant(2)

      return strategy.run(computation)

    expected_result_ones = [1 for _ in range(0, strategy.num_replicas_in_sync)]
    self.assertAllEqual(expected_result_ones,
                        strategy.experimental_local_results(return_one()))

    expected_result_twos = [2 for _ in range(0, strategy.num_replicas_in_sync)]
    self.assertAllEqual(expected_result_twos,
                        strategy.experimental_local_results(return_two()))


if __name__ == "__main__":
  test.main()
