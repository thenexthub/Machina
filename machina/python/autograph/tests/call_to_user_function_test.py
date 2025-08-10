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
"""Calls to dynamic (i.e. nonglobal) functions.

Examples:
 * function variables
 * function parameters
 * factories
"""

import machina as tf

from machina.python.autograph.tests import reference_test_base


def function_1(x):
  return x * x * x


def function_2(x):
  return -1 * x + 11


def factory(n):
  if n == 1:
    return function_1
  return function_2


def static_fn(x):
  a = function_1(x)
  b = function_2(x)
  return a + b


def factory_dynamic_fn(x):
  f = factory(1)
  a = f(x)
  f = factory(2)
  b = f(x)
  return a + b


def param_dynamic_fn(f, x):
  return f(x)


def variable_dynamic_fn(x):
  f = function_1
  a = f(x)
  f = function_2
  b = f(x)
  return a + b


def variable_dynamic_whitelisted_fn(x):
  f = tf.identity
  return f(x)


def dynamic_fn_with_kwargs(f, x):
  return f(x=x)


class ReferenceTest(reference_test_base.TestCase):

  def test_basic(self):
    self.assertFunctionMatchesEager(static_fn, 1)
    self.assertFunctionMatchesEager(factory_dynamic_fn, 1)
    self.assertFunctionMatchesEager(param_dynamic_fn, function_1, 1)
    self.assertFunctionMatchesEager(variable_dynamic_fn, 1)
    self.assertFunctionMatchesEager(variable_dynamic_whitelisted_fn, 1)
    self.assertFunctionMatchesEager(dynamic_fn_with_kwargs, function_1, 1)

  def test_basic_tensor(self):
    self.all_inputs_tensors = True
    self.assertFunctionMatchesEager(static_fn, 1)
    self.assertFunctionMatchesEager(factory_dynamic_fn, 1)
    self.assertFunctionMatchesEager(param_dynamic_fn, function_1, 1)
    self.assertFunctionMatchesEager(variable_dynamic_fn, 1)
    self.assertFunctionMatchesEager(variable_dynamic_whitelisted_fn, 1)
    self.assertFunctionMatchesEager(dynamic_fn_with_kwargs, function_1, 1)


if __name__ == '__main__':
  tf.test.main()
