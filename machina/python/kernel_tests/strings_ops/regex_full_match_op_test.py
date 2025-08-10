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
"""Tests for RegexFullMatch op from string_ops."""

from absl.testing import parameterized

from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.framework import test_util
from machina.python.ops import gen_string_ops
from machina.python.ops import string_ops
from machina.python.platform import test


@parameterized.parameters(
    (gen_string_ops.regex_full_match),
    (gen_string_ops.static_regex_full_match))
class RegexFullMatchOpVariantsTest(test.TestCase, parameterized.TestCase):

  @test_util.run_deprecated_v1
  def testRegexFullMatch(self, op):
    values = ["abaaba", "abcdabcde"]
    with self.cached_session():
      input_tensor = constant_op.constant(values, dtypes.string)
      matched = op(input_tensor, "a.*a").eval()
      self.assertAllEqual([True, False], matched)

  @test_util.run_deprecated_v1
  def testRegexFullMatchTwoDims(self, op):
    values = [["abaaba", "abcdabcde"], ["acdcba", "ebcda"]]
    with self.cached_session():
      input_tensor = constant_op.constant(values, dtypes.string)
      matched = op(input_tensor, "a.*a").eval()
      self.assertAllEqual([[True, False], [True, False]], matched)

  @test_util.run_deprecated_v1
  def testEmptyMatch(self, op):
    values = ["abc", "1"]
    with self.cached_session():
      input_tensor = constant_op.constant(values, dtypes.string)
      matched = op(input_tensor, "").eval()
      self.assertAllEqual([False, False], matched)

  @test_util.run_deprecated_v1
  def testInvalidPattern(self, op):
    values = ["abc", "1"]
    with self.cached_session():
      input_tensor = constant_op.constant(values, dtypes.string)
      invalid_pattern = "A["
      matched = op(input_tensor, invalid_pattern)
      with self.assertRaisesOpError("Invalid pattern"):
        self.evaluate(matched)


class RegexFullMatchOpTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testRegexFullMatchDelegation(self):
    with self.cached_session():
      input_tensor = constant_op.constant("foo", dtypes.string)
      pattern = "[a-z]"
      op = string_ops.regex_full_match(input_tensor, pattern)
      self.assertFalse(op.name.startswith("RegexFullMatch"), op.name)

      pattern_tensor = constant_op.constant("[a-z]*", dtypes.string)
      op_tensor = string_ops.regex_full_match(input_tensor, pattern_tensor)
      self.assertTrue(op_tensor.name.startswith("RegexFullMatch"), op.name)

  @test_util.run_deprecated_v1
  def testStaticRegexFullMatchDelegation(self):
    with self.cached_session():
      input_tensor = constant_op.constant("foo", dtypes.string)
      pattern = "[a-z]*"
      op = string_ops.regex_full_match(input_tensor, pattern)
      self.assertTrue(op.name.startswith("StaticRegexFullMatch"), op.name)

      pattern_tensor = constant_op.constant("[a-z]*", dtypes.string)
      op_vec = string_ops.regex_full_match(input_tensor, pattern_tensor)
      self.assertTrue(op_vec.name.startswith("RegexFullMatch"), op.name)


if __name__ == "__main__":
  test.main()
