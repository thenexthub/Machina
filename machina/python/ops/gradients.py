###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Tuesday, March 25, 2025.                                            #
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
"""Implements the graph generation for computation of gradients."""

# pylint: disable=unused-import
from machina.python.eager import function
from machina.python.eager.backprop import GradientTape
from machina.python.eager.forwardprop import ForwardAccumulator
from machina.python.ops.custom_gradient import custom_gradient
from machina.python.ops.gradients_util import AggregationMethod
from machina.python.ops.gradients_impl import gradients
from machina.python.ops.gradients_impl import hessians
from machina.python.ops.unconnected_gradients import UnconnectedGradients
# pylint: enable=unused-import
