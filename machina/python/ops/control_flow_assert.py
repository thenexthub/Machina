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
"""Assert functions for Control Flow Operations."""

from machina.python.eager import context
from machina.python.framework import dtypes
from machina.python.framework import errors
from machina.python.framework import ops
from machina.python.ops import array_ops
from machina.python.ops import cond
from machina.python.ops import gen_control_flow_ops
from machina.python.ops import gen_logging_ops
from machina.python.ops import gen_math_ops
from machina.python.util import dispatch
from machina.python.util import tf_should_use
from machina.python.util.tf_export import tf_export


def _summarize_eager(tensor, summarize=None):
  """Returns a summarized string representation of eager `tensor`.

  Args:
    tensor: EagerTensor to summarize
    summarize: Include these many first elements of `array`
  """
  # Emulate the behavior of Tensor::SummarizeValue()
  if summarize is None:
    summarize = 3
  elif summarize < 0:
    summarize = array_ops.size(tensor)

  # reshape((-1,)) is the fastest way to get a flat array view
  if tensor._rank():  # pylint: disable=protected-access
    flat = tensor.numpy().reshape((-1,))
    lst = [str(x) for x in flat[:summarize]]
    if len(lst) < flat.size:
      lst.append("...")
  else:
    # tensor.numpy() returns a scalar for zero dimensional arrays
    if gen_math_ops.not_equal(summarize, 0):
      lst = [str(tensor.numpy())]
    else:
      lst = []

  return ", ".join(lst)


# Assert and Print are special symbols in python, so we must
# use an upper-case version of them.
@tf_export("debugging.Assert", "Assert")
@dispatch.add_dispatch_support
@tf_should_use.should_use_result
def Assert(condition, data, summarize=None, name=None):
  """Asserts that the given condition is true.

  If `condition` evaluates to false, print the list of tensors in `data`.
  `summarize` determines how many entries of the tensors to print.

  Args:
    condition: The condition to evaluate.
    data: The tensors to print out when condition is false.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).

  Returns:
    assert_op: An `Operation` that, when executed, raises a
    `tf.errors.InvalidArgumentError` if `condition` is not true.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    @compatibility(TF1)
    When in TF V1 mode (that is, outside `tf.function`) Assert needs a control
    dependency on the output to ensure the assertion executes:

  ```python
  # Ensure maximum element of x is smaller or equal to 1
  assert_op = tf.Assert(tf.less_equal(tf.reduce_max(x), 1.), [x])
  with tf.control_dependencies([assert_op]):
    ... code using x ...
  ```

    @end_compatibility
  """
  if context.executing_eagerly():
    if not condition:
      xs = ops.convert_n_to_tensor(data)
      data_str = [_summarize_eager(x, summarize) for x in xs]
      raise errors.InvalidArgumentError(
          node_def=None,
          op=None,
          message="Expected '%s' to be true. Summarized data: %s" %
          (condition, "\n".join(data_str)))
    return

  with ops.name_scope(name, "Assert", [condition, data]) as name:
    xs = ops.convert_n_to_tensor(data)
    if all(x.dtype in {dtypes.string, dtypes.int32} for x in xs):
      # As a simple heuristic, we assume that string and int32 are
      # on host to avoid the need to use cond. If it is not case,
      # we will pay the price copying the tensor to host memory.
      return gen_logging_ops._assert(condition, data, summarize, name="Assert")  # pylint: disable=protected-access
    else:
      condition = ops.convert_to_tensor(condition, name="Condition")

      def true_assert():
        return gen_logging_ops._assert(  # pylint: disable=protected-access
            condition, data, summarize, name="Assert")

      guarded_assert = cond.cond(
          condition,
          gen_control_flow_ops.no_op,
          true_assert,
          name="AssertGuard")
      if context.executing_eagerly():
        return
      return guarded_assert.op
