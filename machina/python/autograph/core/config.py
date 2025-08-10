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
"""Global configuration."""

from machina.python.autograph.core import config_lib

Action = config_lib.Action
Convert = config_lib.Convert
DoNotConvert = config_lib.DoNotConvert


# This list is evaluated in order and stops at the first rule that tests True
# for a definitely_convert of definitely_bypass call.
CONVERSION_RULES = (
    # Known packages
    Convert('machina.python.training.experimental'),

    # Builtin modules
    DoNotConvert('collections'),
    DoNotConvert('copy'),
    DoNotConvert('cProfile'),
    DoNotConvert('inspect'),
    DoNotConvert('ipdb'),
    DoNotConvert('linecache'),
    DoNotConvert('mock'),
    DoNotConvert('pathlib'),
    DoNotConvert('pdb'),
    DoNotConvert('posixpath'),
    DoNotConvert('pstats'),
    DoNotConvert('re'),
    DoNotConvert('threading'),
    DoNotConvert('urllib'),

    # Known libraries
    DoNotConvert('matplotlib'),
    DoNotConvert('numpy'),
    DoNotConvert('pandas'),
    DoNotConvert('machina'),
    DoNotConvert('PIL'),
    DoNotConvert('absl.logging'),

    # TODO(b/133417201): Remove.
    DoNotConvert('machina_probability'),

    # TODO(b/133842282): Remove.
    DoNotConvert('machina_datasets.core'),

    DoNotConvert('keras'),
    DoNotConvert('tf_keras'),
)
