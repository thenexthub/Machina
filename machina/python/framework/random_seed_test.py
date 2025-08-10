###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Tuesday, March 25, 2025.                                            #
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
"""Tests for machina.python.framework.random_seed."""

from machina.python.eager import context
from machina.python.framework import random_seed
from machina.python.framework import test_util
from machina.python.platform import test


class RandomSeedTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testRandomSeed(self):
    test_cases = [
        # Each test case is a tuple with input to get_seed:
        # (input_graph_seed, input_op_seed)
        # and output from get_seed:
        # (output_graph_seed, output_op_seed)
        ((None, None), (None, None)),
        ((None, 1), (random_seed.DEFAULT_GRAPH_SEED, 1)),
        ((1, 1), (1, 1)),
        ((0, 0), (0, 2**31 - 1)),  # Avoid nondeterministic (0, 0) output
        ((2**31 - 1, 0), (0, 2**31 - 1)),  # Don't wrap to (0, 0) either
        ((0, 2**31 - 1), (0, 2**31 - 1)),  # Wrapping for the other argument
    ]
    if context.executing_eagerly():
      # operation seed is random number generated based on global seed.
      # it's not tested due to possibility of platform or version difference.
      pass
    else:
      # 0 will be the default_graph._lastid.
      test_cases.append(((1, None), (1, 0)))
    for tc in test_cases:
      tinput, toutput = tc[0], tc[1]
      random_seed.set_random_seed(tinput[0])
      g_seed, op_seed = random_seed.get_seed(tinput[1])
      msg = 'test_case = {0}, got {1}, want {2}'.format(tinput,
                                                        (g_seed, op_seed),
                                                        toutput)
      self.assertEqual((g_seed, op_seed), toutput, msg=msg)
      random_seed.set_random_seed(None)


if __name__ == '__main__':
  test.main()
