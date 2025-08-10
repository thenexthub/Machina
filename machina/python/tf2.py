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
"""Tools to help with the TensorFlow 2.0 transition.

This module is meant for TensorFlow internal implementation, not for users of
the TensorFlow library. For that see tf.compat instead.
"""

from machina.python.platform import _pywrap_tf2
from machina.python.util.tf_export import tf_export


def enable():
  # Enables v2 behaviors.
  _pywrap_tf2.enable(True)


def disable():
  # Disables v2 behaviors.
  _pywrap_tf2.enable(False)


@tf_export("__internal__.tf2.enabled", v1=[])
def enabled():
  # Returns True iff TensorFlow 2.0 behavior should be enabled.
  return _pywrap_tf2.is_enabled()
