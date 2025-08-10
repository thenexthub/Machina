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
"""Common utilities and settings used by tfdbg v2's op callbacks."""

# The ops that are skipped by tfdbg v2's op callbacks.
# They belong to TensorFlow's control flow ops (e.g., "Enter", "StatelessIf")
# and ops that wrap nested tf.function calls.
OP_CALLBACK_SKIP_OPS = (
    # TODO(b/139668453): The following skipped ops are related to a limitation
    # in the op callback.
    b"Enter",
    b"Exit",
    b"Identity",
    b"If",
    b"LoopCond",
    b"Merge",
    b"NextIteration",
    b"StatelessIf",
    b"StatefulPartitionedCall",
    b"Switch",
    b"While",
    # NOTE(b/154097452): On TPUs, debugger ops are colocated with RemoteCall
    # ops. This exclusion prevents an error due to no OpKernel for those
    # debugger ops.
    b"RemoteCall",
    # TPU-specific ops begin.
    b"TPUReplicatedInput",
    b"TPUReplicateMetadata",
    b"TPUCompilationResult",
    b"TPUReplicatedOutput",
    b"ConfigureDistributedTPU",
    # Other special ops used by TensorFlow internally.
    b"DestroyResourceOp",
)
