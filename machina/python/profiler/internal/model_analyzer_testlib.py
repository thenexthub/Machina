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
"""A test lib that defines some models."""
import contextlib

from machina.python.framework import dtypes
from machina.python.ops import array_ops
from machina.python.ops import init_ops
from machina.python.ops import math_ops
from machina.python.ops import nn_grad  # pylint: disable=unused-import
from machina.python.ops import nn_ops
from machina.python.ops import rnn
from machina.python.ops import rnn_cell
from machina.python.ops import tensor_array_grad  # pylint: disable=unused-import
from machina.python.ops import variable_scope
from machina.python.profiler import model_analyzer
from machina.python.training import gradient_descent
from machina.python.util import _pywrap_tfprof as print_mdl
from machina.python.util import compat


def BuildSmallModel():
  """Build a small forward conv model."""
  image = array_ops.zeros([2, 6, 6, 3])
  _ = variable_scope.get_variable(
      'ScalarW', [],
      dtypes.float32,
      initializer=init_ops.random_normal_initializer(stddev=0.001))
  kernel = variable_scope.get_variable(
      'DW', [3, 3, 3, 6],
      dtypes.float32,
      initializer=init_ops.random_normal_initializer(stddev=0.001))
  x = nn_ops.conv2d(image, kernel, [1, 2, 2, 1], padding='SAME')
  kernel = variable_scope.get_variable(
      'DW2', [2, 2, 6, 12],
      dtypes.float32,
      initializer=init_ops.random_normal_initializer(stddev=0.001))
  x = nn_ops.conv2d(x, kernel, [1, 2, 2, 1], padding='SAME')
  return x


def BuildFullModel():
  """Build the full model with conv,rnn,opt."""
  seq = []
  for i in range(4):
    with variable_scope.variable_scope('inp_%d' % i):
      seq.append(array_ops.reshape(BuildSmallModel(), [2, 1, -1]))

  cell = rnn_cell.BasicRNNCell(16)
  out = rnn.dynamic_rnn(
      cell, array_ops.concat(seq, axis=1), dtype=dtypes.float32)[0]

  target = array_ops.ones_like(out)
  loss = nn_ops.l2_loss(math_ops.reduce_mean(target - out))
  sgd_op = gradient_descent.GradientDescentOptimizer(1e-2)
  return sgd_op.minimize(loss)


def BuildSplittableModel():
  """Build a small model that can be run partially in each step."""
  image = array_ops.zeros([2, 6, 6, 3])

  kernel1 = variable_scope.get_variable(
      'DW', [3, 3, 3, 6],
      dtypes.float32,
      initializer=init_ops.random_normal_initializer(stddev=0.001))
  r1 = nn_ops.conv2d(image, kernel1, [1, 2, 2, 1], padding='SAME')

  kernel2 = variable_scope.get_variable(
      'DW2', [2, 3, 3, 6],
      dtypes.float32,
      initializer=init_ops.random_normal_initializer(stddev=0.001))
  r2 = nn_ops.conv2d(image, kernel2, [1, 2, 2, 1], padding='SAME')

  r3 = r1 + r2
  return r1, r2, r3


def SearchTFProfNode(node, name):
  """Search a node in the tree."""
  if node.name == name:
    return node
  for c in node.children:
    r = SearchTFProfNode(c, name)
    if r: return r
  return None


@contextlib.contextmanager
def ProfilerFromFile(profile_file):
  """Initialize a profiler from profile file."""
  print_mdl.ProfilerFromFile(compat.as_bytes(profile_file))
  profiler = model_analyzer.Profiler.__new__(model_analyzer.Profiler)
  yield profiler
  print_mdl.DeleteProfiler()


def CheckAndRemoveDoc(profile):
  assert 'Doc:' in profile
  start_pos = profile.find('Profile:')
  return profile[start_pos + 9:]
