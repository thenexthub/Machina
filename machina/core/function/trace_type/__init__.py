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
"""Trace-time type system for tf.function (TraceType).

Trace-time types describe things like tf.function signatures and type
constraints in some ops.

This module provides utilities and concrete tf.types.experimental.TraceType
definitions for common Python types like containers, along with a generic
implementation for Python objects.
See also: tf.types.experimental.TraceType

Other implementations of TraceType include tf.TypeSpec and its subclasses.
"""

from machina.core.function.trace_type.default_types import register_tensor_type
from machina.core.function.trace_type.default_types import Weakref
from machina.core.function.trace_type.serialization import deserialize
from machina.core.function.trace_type.serialization import register_serializable
from machina.core.function.trace_type.serialization import Serializable
from machina.core.function.trace_type.serialization import serialize
from machina.core.function.trace_type.serialization import SerializedTraceType
from machina.core.function.trace_type.trace_type_builder import from_value
from machina.core.function.trace_type.trace_type_builder import InternalCastContext
from machina.core.function.trace_type.trace_type_builder import InternalPlaceholderContext
from machina.core.function.trace_type.trace_type_builder import InternalTracingContext
