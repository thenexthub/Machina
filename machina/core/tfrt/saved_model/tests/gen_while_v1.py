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
"""Generates a toy v1 saved model for testing."""

import shutil
from absl import app
from absl import flags
from machina.python.client import session
from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.ops import array_ops
from machina.python.ops import math_ops
from machina.python.ops import while_loop
from machina.python.saved_model import builder
from machina.python.saved_model import signature_constants
from machina.python.saved_model import signature_def_utils
from machina.python.saved_model import tag_constants
from machina.python.saved_model import utils

flags.DEFINE_string('saved_model_path', '', 'Path to save the model to.')
FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  shutil.rmtree(FLAGS.saved_model_path)

  # Create the graph
  ten = constant_op.constant(10)
  one = constant_op.constant(1)
  x = array_ops.placeholder(dtypes.int32, shape=(), name='input')
  r = while_loop.while_loop(lambda a: a < ten, lambda a: math_ops.add(a, one),
                            [x])

  sess = session.Session()

  sm_builder = builder.SavedModelBuilder(FLAGS.saved_model_path)
  tensor_info_x = utils.build_tensor_info(x)
  tensor_info_r = utils.build_tensor_info(r)

  func_signature = (
      signature_def_utils.build_signature_def(
          inputs={'x': tensor_info_x},
          outputs={'r': tensor_info_r},
          method_name=signature_constants.PREDICT_METHOD_NAME))

  sm_builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
          'serving_default': func_signature,
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: func_signature,
      },
      strip_default_attrs=True)
  sm_builder.save()


if __name__ == '__main__':
  app.run(main)
