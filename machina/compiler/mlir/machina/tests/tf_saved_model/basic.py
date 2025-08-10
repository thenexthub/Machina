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

# RUN: %p/basic | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import machina.compat.v2 as tf
from machina.compiler.mlir.machina.tests.tf_saved_model import common


# Verify that the tf.versions attribute exists. It is difficult to enforce
# contents, since the version numbers change over time. The conversion logic
# itself is verified in the common graphdef converter, so here just assert
# it is being invoked.
# CHECK: module
# CHECK-SAME: tf.versions
# CHECK-SAME: bad_consumers
# CHECK-SAME: min_consumer
# CHECK-SAME: producer


class TestModule(tf.Module):

  def __init__(self):
    super(TestModule, self).__init__()
    self.v42 = tf.Variable(42.0)
    self.c43 = tf.constant(43.0)

  # During serialization, the constants are given internal (non-user-accessible, non-semantically-load-bearing) exported names.
  # CHECK: "tf_saved_model.global_tensor"() <{sym_name = "[[CONST:[a-zA-Z_0-9.]+]]", type = tensor<f32>, value = dense<4.300000e+01> : tensor<f32>}> {tf_saved_model.exported_names = [{{.*}}]} : () -> ()

  # CHECK: "tf_saved_model.global_tensor"() <{is_mutable, sym_name = "[[VAR:[a-zA-Z_0-9]+]]", type = tensor<f32>, value = dense<4.200000e+01> : tensor<f32>}> {tf_saved_model.exported_names = ["v42"]} : () -> ()
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<f32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]},
  # CHECK-SAME:   %arg1: tensor<!tf_type.resource<tensor<f32>>>
  # CHECK-SAME:   %arg2: tensor<!tf_type.resource<tensor<f32>>>
  # CHECK-SAME:   tensor<f32> {tf_saved_model.index_path = []})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["some_function"]
  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def some_function(self, x):
    return x + self.v42 + self.c43


if __name__ == '__main__':
  common.do_test(TestModule)
