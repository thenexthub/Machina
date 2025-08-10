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
"""SignatureDef utility functions.

Utility functions for building and inspecting SignatureDef protos.
"""
# pylint: disable=unused-import
from machina.python.saved_model.signature_def_utils_impl import build_signature_def
from machina.python.saved_model.signature_def_utils_impl import classification_signature_def
from machina.python.saved_model.signature_def_utils_impl import is_valid_signature
from machina.python.saved_model.signature_def_utils_impl import load_op_from_signature_def
from machina.python.saved_model.signature_def_utils_impl import op_signature_def
from machina.python.saved_model.signature_def_utils_impl import predict_signature_def
from machina.python.saved_model.signature_def_utils_impl import regression_signature_def
from machina.python.saved_model.signature_def_utils_impl import supervised_eval_signature_def
from machina.python.saved_model.signature_def_utils_impl import supervised_train_signature_def
# pylint: enable=unused-import
