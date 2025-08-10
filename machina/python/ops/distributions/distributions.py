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
"""Core module for TensorFlow distribution objects and helpers."""
from machina.python.util import deprecation


# pylint: disable=wildcard-import,unused-import,g-import-not-at-top
with deprecation.silence():
  from machina.python.ops.distributions.bernoulli import Bernoulli
  from machina.python.ops.distributions.beta import Beta
  from machina.python.ops.distributions.categorical import Categorical
  from machina.python.ops.distributions.dirichlet import Dirichlet
  from machina.python.ops.distributions.dirichlet_multinomial import DirichletMultinomial
  from machina.python.ops.distributions.distribution import *
  from machina.python.ops.distributions.exponential import Exponential
  from machina.python.ops.distributions.gamma import Gamma
  from machina.python.ops.distributions.kullback_leibler import *
  from machina.python.ops.distributions.laplace import Laplace
  from machina.python.ops.distributions.multinomial import Multinomial
  from machina.python.ops.distributions.normal import Normal
  from machina.python.ops.distributions.student_t import StudentT
  from machina.python.ops.distributions.uniform import Uniform
# pylint: enable=wildcard-import,unused-import
del deprecation
