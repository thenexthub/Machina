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

from machina.python.client import session
from machina.python.eager import backprop
from machina.python.eager import def_function
from machina.python.framework import dtypes
from machina.python.framework import errors
from machina.python.framework import ops
from machina.python.ops import array_ops
from machina.python.ops import string_ops
from machina.python.platform import test


class JitCompileTest(test.TestCase):

  def testBasic(self):
    with ops.Graph().as_default() as g:

      def fn(x, a):
        return x + a

      xla_func = def_function.function(fn, jit_compile=True)
      inputs = array_ops.placeholder(dtypes.float32, [5])
      x = xla_func(inputs, 1)
      with session.Session(graph=g) as sess:
        y = sess.run(x, feed_dict={inputs: [1, 2, 2, 3, 3]})
        self.assertTrue(x.graph.as_graph_def().library.function[0]
                        .attr["_XlaMustCompile"].b)
        self.assertAllClose([2, 3, 3, 4, 4], y)

  def testDerivative(self):
    def fn(x, a):
      return 2 * x + a

    with ops.Graph().as_default() as g:
      xla_func = def_function.function(fn, jit_compile=True)
      with backprop.GradientTape() as tape:
        inputs = array_ops.placeholder(dtypes.float32, [5])
        tape.watch(inputs)
        outputs = xla_func(inputs, 1)
      grads = tape.gradient(outputs, inputs)

    with session.Session(graph=g) as sess:
      grads_tensor = sess.run(grads, feed_dict={inputs: [1, 2, 2, 3, 3]})
      self.assertAllClose([2, 2, 2, 2, 2], grads_tensor)
      (forward, backward) = xla_func.get_concrete_function(
          inputs, 1)._delayed_rewrite_functions.forward_backward()

      # Check that the must-compile attribute gets correctly propagated to the
      # created derivatives.
      self.assertTrue(forward.cached_definition.attr["_XlaMustCompile"])
      self.assertTrue(backward.function_def.attr["_XlaMustCompile"])

  def testBasicInt32(self):
    with ops.Graph().as_default() as g:

      def fn(x, a):
        return x + a

      xla_func = def_function.function(fn, jit_compile=True)
      inputs = array_ops.placeholder(dtypes.int32, [5])
      x = xla_func(inputs, 1)
      with session.Session(graph=g) as sess:
        y = sess.run(x, feed_dict={inputs: [1, 2, 2, 3, 3]})
        self.assertTrue(x.graph.as_graph_def().library.function[0]
                        .attr["_XlaMustCompile"].b)
        self.assertAllClose([2, 3, 3, 4, 4], y)

  # Checking that we crash on an unsupported operation lets us test that the XLA
  # compiler was actually invoked.
  def testUnsupportedOps(self):
    with ops.Graph().as_default() as g:

      def fn(x):
        return string_ops.string_length(
            string_ops.string_format('{}', x))

      xla_func = def_function.function(fn, jit_compile=True)
      inputs = array_ops.placeholder(dtypes.float32, [5])
      x = xla_func(inputs)
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Detected unsupported operations"):
        with session.Session(graph=g) as sess:
          sess.run(x, feed_dict={inputs: [1, 2, 2, 3, 3]})


if __name__ == "__main__":
  test.main()
