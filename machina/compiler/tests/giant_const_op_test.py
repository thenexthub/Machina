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
"""Tests for giant const op compilation."""

import os
import numpy as np

from machina.python.distribute import tpu_strategy as tpu_lib
from machina.python.distribute.cluster_resolver import tpu_cluster_resolver
from machina.python.eager import def_function
from machina.python.eager import remote
from machina.python.eager import test
from machina.python.framework import config
from machina.python.framework import constant_op
from machina.python.framework import dtypes
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
  return tpu_lib.TPUStrategyV2(resolver)


# This test doesn't use XLATestCase like the other tests in this directory.
# The Const op xla op kernel is compilation only and therefore is not executed
# with XLA in the on demand compilation mode. Also, here we want to feed the
# full program to XLA to verify handling of programs with giant constant
# tensors.
class GiantConstOp(test.TestCase):

  # Verifies that graphs containing giant const tensors that won't fit in memory
  # are compiled correctly to HLO.
  def testGiantConst(self):
    # Disabling Mlir bridge since using the tf2xla implementation of
    # StridedSliceop which would get executed in this GiantConst test.
    config.disable_mlir_bridge()
    strategy = get_tpu_strategy()
    types = {
        dtypes.bool,
        dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
        dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64,
        dtypes.float16, dtypes.bfloat16,
        dtypes.float32, dtypes.float64,
    }
    for dtype in types:
      values = [True if dtype is dtypes.bool else 1]

      if dtype is dtypes.bool:
        values.append(False)
      elif dtype is not dtypes.float64:
        # TPUs don't follow IEEE 754 float64 standard for 64 bit floating point
        # numbers so it could return different output even with just data
        # transformation ops without any arithmetic operations.
        values.extend([dtype.min, dtype.max])

      for value in values:

        @def_function.function
        def train_step():

          # pylint: disable=cell-var-from-loop
          def computation():
            const = constant_op.constant(value, dtype=dtype, shape=[1024]*4)
            return const[:1, :1, :1, :1]

          return strategy.run(computation, args=())

        output = strategy.experimental_local_results(train_step())[0]
        expected = np.full((1, 1, 1, 1), value)
        self.assertAllEqual(output, expected)

if __name__ == "__main__":
  # Make sure TF_MACHINA_MACHINA_XLA_FLAGS is not already set to avoid dropping the existing
  # value silently.
  assert "TF_MACHINA_MACHINA_XLA_FLAGS" not in os.environ

  # Disable tfxla constant folding that always creates full Tensors and will
  # fail for giant tensors.
  os.environ["TF_MACHINA_MACHINA_XLA_FLAGS"] = "--tf_xla_disable_constant_folding=true"

  test.main()
