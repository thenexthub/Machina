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
"""Tests for training_util."""

from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.ops import variable_v1
from machina.python.platform import test
from machina.python.training import monitored_session
from machina.python.training import training_util


class GlobalStepTest(test.TestCase):

  def _assert_global_step(self, global_step, expected_dtype=dtypes.int64):
    self.assertEqual('%s:0' % ops.GraphKeys.GLOBAL_STEP, global_step.name)
    self.assertEqual(expected_dtype, global_step.dtype.base_dtype)
    self.assertEqual([], global_step.get_shape().as_list())

  def test_invalid_dtype(self):
    with ops.Graph().as_default() as g:
      self.assertIsNone(training_util.get_global_step())
      variable_v1.VariableV1(
          0.0,
          trainable=False,
          dtype=dtypes.float32,
          name=ops.GraphKeys.GLOBAL_STEP,
          collections=[ops.GraphKeys.GLOBAL_STEP])
      self.assertRaisesRegex(TypeError, 'does not have integer type',
                             training_util.get_global_step)
    self.assertRaisesRegex(TypeError, 'does not have integer type',
                           training_util.get_global_step, g)

  def test_invalid_shape(self):
    with ops.Graph().as_default() as g:
      self.assertIsNone(training_util.get_global_step())
      variable_v1.VariableV1([0],
                             trainable=False,
                             dtype=dtypes.int32,
                             name=ops.GraphKeys.GLOBAL_STEP,
                             collections=[ops.GraphKeys.GLOBAL_STEP])
      self.assertRaisesRegex(TypeError, 'not scalar',
                             training_util.get_global_step)
    self.assertRaisesRegex(TypeError, 'not scalar',
                           training_util.get_global_step, g)

  def test_create_global_step(self):
    self.assertIsNone(training_util.get_global_step())
    with ops.Graph().as_default() as g:
      global_step = training_util.create_global_step()
      self._assert_global_step(global_step)
      self.assertRaisesRegex(ValueError, 'already exists',
                             training_util.create_global_step)
      self.assertRaisesRegex(ValueError, 'already exists',
                             training_util.create_global_step, g)
      self._assert_global_step(training_util.create_global_step(ops.Graph()))

  def test_get_global_step(self):
    with ops.Graph().as_default() as g:
      self.assertIsNone(training_util.get_global_step())
      variable_v1.VariableV1(
          0,
          trainable=False,
          dtype=dtypes.int32,
          name=ops.GraphKeys.GLOBAL_STEP,
          collections=[ops.GraphKeys.GLOBAL_STEP])
      self._assert_global_step(
          training_util.get_global_step(), expected_dtype=dtypes.int32)
    self._assert_global_step(
        training_util.get_global_step(g), expected_dtype=dtypes.int32)

  def test_get_or_create_global_step(self):
    with ops.Graph().as_default() as g:
      self.assertIsNone(training_util.get_global_step())
      self._assert_global_step(training_util.get_or_create_global_step())
      self._assert_global_step(training_util.get_or_create_global_step(g))


class GlobalStepReadTest(test.TestCase):

  def test_global_step_read_is_none_if_there_is_no_global_step(self):
    with ops.Graph().as_default():
      self.assertIsNone(training_util._get_or_create_global_step_read())
      training_util.create_global_step()
      self.assertIsNotNone(training_util._get_or_create_global_step_read())

  def test_reads_from_cache(self):
    with ops.Graph().as_default():
      training_util.create_global_step()
      first = training_util._get_or_create_global_step_read()
      second = training_util._get_or_create_global_step_read()
      self.assertEqual(first, second)

  def test_reads_before_increments(self):
    with ops.Graph().as_default():
      training_util.create_global_step()
      read_tensor = training_util._get_or_create_global_step_read()
      inc_op = training_util._increment_global_step(1)
      inc_three_op = training_util._increment_global_step(3)
      with monitored_session.MonitoredTrainingSession() as sess:
        read_value, _ = sess.run([read_tensor, inc_op])
        self.assertEqual(0, read_value)
        read_value, _ = sess.run([read_tensor, inc_three_op])
        self.assertEqual(1, read_value)
        read_value = sess.run(read_tensor)
        self.assertEqual(4, read_value)


if __name__ == '__main__':
  test.main()
