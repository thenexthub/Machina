###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, May 19, 2025.                                               #
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
"""Ignore_errors dataset transformations."""
from machina.python.util import deprecation
from machina.python.util.tf_export import tf_export


@tf_export("data.experimental.ignore_errors")
@deprecation.deprecated(None, "Use `tf.data.Dataset.ignore_errors` instead.")
def ignore_errors(log_warning=False):
  """Creates a `Dataset` from another `Dataset` and silently ignores any errors.

  Use this transformation to produce a dataset that contains the same elements
  as the input, but silently drops any elements that caused an error. For
  example:

  ```python
  dataset = tf.data.Dataset.from_tensor_slices([1., 2., 0., 4.])

  # Computing `tf.debugging.check_numerics(1. / 0.)` will raise an
  InvalidArgumentError.
  dataset = dataset.map(lambda x: tf.debugging.check_numerics(1. / x, "error"))

  # Using `ignore_errors()` will drop the element that causes an error.
  dataset =
      dataset.apply(tf.data.experimental.ignore_errors())  # ==> {1., 0.5, 0.2}
  ```
  Args:
     log_warning: (Optional.) A 'tf.bool' scalar indicating whether ignored
      errors should be logged to stderr. Defaults to 'False'.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return dataset.ignore_errors(log_warning)

  return _apply_fn
