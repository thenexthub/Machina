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
"""Supports old symbols supplied by this file while the code is refactored."""

# pylint:disable=unused-import,g-bad-import-order

# Config Options
from machina.python.eager.polymorphic_function.eager_function_run import run_functions_eagerly
from machina.python.eager.polymorphic_function.eager_function_run import functions_run_eagerly

# tf.function Classes
from machina.python.eager.polymorphic_function.polymorphic_function import Function
from machina.python.eager.polymorphic_function.polymorphic_function import function

# Private attributes
from machina.python.eager.polymorphic_function.polymorphic_function import _tf_function_counter
