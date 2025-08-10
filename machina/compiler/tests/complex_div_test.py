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
"""Test cases for complex numbers division."""

import os

import numpy as np

from machina.compiler.tests import xla_test
from machina.python.framework import dtypes
from machina.python.ops import array_ops
from machina.python.ops import gen_math_ops
from machina.python.platform import googletest

os.environ["MACHINA_MACHINA_XLA_FLAGS"] = ("--xla_cpu_fast_math_honor_nans=true "
                           "--xla_cpu_fast_math_honor_infs=true")


class ComplexNumbersDivisionTest(xla_test.XLATestCase):
  """Test cases for complex numbers division operators."""

  def _testBinary(self, op, a, b, expected, equality_test=None):
    with self.session() as session:
      with self.test_scope():
        pa = array_ops.placeholder(dtypes.as_dtype(a.dtype), a.shape, name="a")
        pb = array_ops.placeholder(dtypes.as_dtype(b.dtype), b.shape, name="b")
        output = op(pa, pb)
      result = session.run(output, {pa: a, pb: b})
      if equality_test is None:
        equality_test = self.assertAllCloseAccordingToType
      equality_test(np.real(result), np.real(expected), rtol=1e-3)
      equality_test(np.imag(result), np.imag(expected), rtol=1e-3)

  def testComplexOps(self):
    for dtype in self.complex_types:
      # Test division by 0 scenarios.
      self._testBinary(
          gen_math_ops.real_div,
          np.array([
              complex(1, 1),
              complex(1, np.inf),
              complex(1, np.nan),
              complex(np.inf, 1),
              complex(np.inf, np.inf),
              complex(np.inf, np.nan),
              complex(np.nan, 1),
              complex(np.nan, np.inf),
              complex(np.nan, np.nan),
              complex(-np.inf, np.nan),
          ],
                   dtype=dtype),
          np.array([
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0 + 0j,
              0.0 + 0j,
          ],
                   dtype=dtype),
          expected=np.array([
              complex(np.inf, np.inf),
              complex(np.inf, np.inf),
              complex(np.inf, np.nan),
              complex(np.inf, np.inf),
              complex(np.inf, np.inf),
              complex(np.inf, np.nan),
              complex(np.nan, np.inf),
              complex(np.nan, np.inf),
              complex(np.nan, np.nan),
              complex(-np.inf, np.nan),
          ],
                            dtype=dtype))

      # Test division with finite numerator, inf/nan denominator.
      self._testBinary(
          gen_math_ops.real_div,
          np.array([
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
          ],
                   dtype=dtype),
          np.array(
              [
                  complex(1, np.inf),
                  complex(1, np.nan),
                  complex(np.inf, 1),
                  complex(np.inf, np.inf),  # C++ and Python diverge here.
                  complex(np.inf, np.nan),  # C++ and Python diverge here.
                  complex(np.nan, 1),
                  complex(np.nan, np.inf),  # C++ and Python diverge here.
                  complex(np.nan, -np.inf),  # C++ and Python diverge here.
                  complex(np.nan, np.nan),
              ],
              dtype=dtype),
          expected=np.array(
              [
                  (1 + 1j) / complex(1, np.inf),
                  (1 + 1j) / complex(1, np.nan),
                  (1 + 1j) / complex(np.inf, 1),
                  complex(0 + 0j),  # C++ and Python diverge here.
                  complex(0 + 0j),  # C++ and Python diverge here.
                  (1 + 1j) / complex(np.nan, 1),
                  complex(0 + 0j),  # C++ and Python diverge here.
                  complex(0 - 0j),  # C++ and Python diverge here.
                  (1 + 1j) / complex(np.nan, np.nan),
              ],
              dtype=dtype))

      # Test division with inf/nan numerator, infinite denominator.
      self._testBinary(
          gen_math_ops.real_div,
          np.array([
              complex(1, np.inf),
              complex(1, np.nan),
              complex(np.inf, 1),
              complex(np.inf, np.inf),
              complex(np.inf, np.nan),
              complex(np.nan, 1),
              complex(np.nan, np.inf),
              complex(np.nan, np.nan),
              complex(np.nan, -np.inf),
          ],
                   dtype=dtype),
          np.array([
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              1 + 1j,
              -1 - 1j,
          ],
                   dtype=dtype),
          expected=np.array(
              [
                  complex(np.inf, np.inf),  # C++ and Python diverge here.
                  complex(1 / np.nan) / (1 + 1j),
                  complex(np.inf / 1) / (1 + 1j),
                  complex(np.inf, -np.nan),  # C++ and Python diverge here.
                  complex(np.inf, -np.inf),  # C++ and Python diverge here.
                  complex(np.nan / 1) / (1 + 1j),
                  complex(np.inf, np.inf),  # C++ and Python diverge here.
                  complex(np.nan / np.nan) / (1 + 1j),
                  complex(np.inf, np.inf),  # C++ and Python diverge here.
              ],
              dtype=dtype))


if __name__ == "__main__":
  googletest.main()
