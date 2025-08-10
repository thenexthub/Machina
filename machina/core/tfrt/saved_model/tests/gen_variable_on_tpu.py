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
"""Generates a toy v2 saved model for testing."""

from absl import app
from absl import flags
from absl import logging

from machina.python.compat import v2_compat
from machina.python.eager import def_function
from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.framework import ops
from machina.python.framework import tensor_spec
from machina.python.module import module
from machina.python.ops import math_ops
from machina.python.ops import variables
from machina.python.saved_model import save
from machina.python.saved_model import save_options

flags.DEFINE_string('saved_model_path', '', 'Path to save the model to.')
FLAGS = flags.FLAGS


class ToyModule(module.Module):
  """Defines a toy module."""

  def __init__(self):
    super(ToyModule, self).__init__()
    self.w = variables.Variable(constant_op.constant([[1], [2], [3]]), name='w')

  @def_function.function(input_signature=[
      tensor_spec.TensorSpec([1, 3], dtypes.int32, name='input')
  ])
  def toy(self, x):
    with ops.device('/device:TPU:0'):
      w = self.w.read_value()
    r = math_ops.matmul(x, w, name='result')
    return r


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  v2_compat.enable_v2_behavior()

  save.save(
      ToyModule(),
      FLAGS.saved_model_path,
      options=save_options.SaveOptions(save_debug_info=False))
  logging.info('Saved model to: %s', FLAGS.saved_model_path)


if __name__ == '__main__':
  app.run(main)
