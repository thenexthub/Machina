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

# RUN: %p/debug_info | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import machina.compat.v2 as tf
from machina.compiler.mlir.machina.tests.tf_saved_model import common


class TestModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.float32)
  ])
  def some_function(self, x, y):
    return x + y
    # Basic check that the debug info file is being correctly saved and loaded.
    #
    # CHECK: "tf.AddV2"{{.*}}loc(#loc{{[0-9]+}})
    # CHECK: "tf.Identity"{{.*}}loc(#loc{{[0-9]+}})
    # CHECK: #loc{{[0-9]+}} = loc("{{.*}}debug_info.py":{{[0-9]+}}:{{[0-9]+}})
    # CHECK: #loc{{[0-9]+}} = loc(callsite(#loc{{[0-9]+}} at #loc{{[0-9]+}}))


if __name__ == '__main__':
  common.do_test(TestModule, show_debug_info=True)
