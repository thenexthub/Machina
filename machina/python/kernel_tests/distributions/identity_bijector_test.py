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
"""Identity Tests."""

from machina.python.framework import test_util
from machina.python.ops.distributions import bijector_test_util
from machina.python.ops.distributions import identity_bijector
from machina.python.platform import test


class IdentityBijectorTest(test.TestCase):
  """Tests correctness of the Y = g(X) = X transformation."""

  def testBijector(self):
    bijector = identity_bijector.Identity(validate_args=True)
    self.assertEqual("identity", bijector.name)
    x = [[[0.], [1.]]]
    self.assertAllEqual(x, self.evaluate(bijector.forward(x)))
    self.assertAllEqual(x, self.evaluate(bijector.inverse(x)))
    self.assertAllEqual(
        0.,
        self.evaluate(
            bijector.inverse_log_det_jacobian(x, event_ndims=3)))
    self.assertAllEqual(
        0.,
        self.evaluate(
            bijector.forward_log_det_jacobian(x, event_ndims=3)))

  @test_util.run_deprecated_v1
  def testScalarCongruency(self):
    with self.cached_session():
      bijector = identity_bijector.Identity()
      bijector_test_util.assert_scalar_congruency(
          bijector, lower_x=-2., upper_x=2.)


if __name__ == "__main__":
  test.main()
