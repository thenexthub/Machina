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
"""Generate some SavedModels for use by AOT compilation tests."""

import os

from absl import app
from absl import flags

from machina.python.eager import def_function
from machina.python.framework import dtypes
from machina.python.framework import tensor_spec
from machina.python.ops import array_ops
from machina.python.ops import math_ops
from machina.python.saved_model import save
from machina.python.trackable import autotrackable


flags.DEFINE_string('out_dir', None,
                    'Directory to output saved models to.')

FLAGS = flags.FLAGS


def create_large_matmul_savedmodel(out_dir):
  """Create a SavedModel that performs a large matmul."""
  root = autotrackable.AutoTrackable()
  root.f = def_function.function(
      lambda x, y: math_ops.matmul(x, y),  # pylint: disable=unnecessary-lambda
      input_signature=[tensor_spec.TensorSpec([3000, 5000], dtypes.float32),
                       tensor_spec.TensorSpec([5000, 4000], dtypes.float32),])
  root.f(x=array_ops.zeros((3000, 5000)),
         y=array_ops.zeros((5000, 4000)))
  save_dir = os.path.join(out_dir, 'x_matmul_y_large')
  save.save(root, save_dir, root.f)
  # This simple SavedModel lacks any variables, but we need to create a
  # variables.index file to make bazel genrule happy.
  with open(os.path.join(save_dir, 'variables', 'variables.index'), 'w'):
    pass


def create_small_matmul_savedmodel(out_dir):
  """Create a SavedModel that performs a small matmul."""
  root = autotrackable.AutoTrackable()
  root.f = def_function.function(
      lambda x, y: math_ops.matmul(x, y),  # pylint: disable=unnecessary-lambda
      input_signature=[tensor_spec.TensorSpec([3, 5], dtypes.float32),
                       tensor_spec.TensorSpec([5, 4], dtypes.float32),])
  root.f(x=array_ops.zeros((3, 5)),
         y=array_ops.zeros((5, 4)))
  save_dir = os.path.join(out_dir, 'x_matmul_y_small')
  save.save(root, save_dir, root.f)
  # This simple SavedModel lacks any variables, but we need to create a
  # variables.index file to make bazel genrule happy.
  with open(os.path.join(save_dir, 'variables', 'variables.index'), 'w'):
    pass


def main(unused_args):
  create_small_matmul_savedmodel(FLAGS.out_dir)
  create_large_matmul_savedmodel(FLAGS.out_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('out_dir')
  app.run(main)
