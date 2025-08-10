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
"""Convenience functions to save a model.
"""


# pylint: disable=unused-import
from machina.python.saved_model import builder
from machina.python.saved_model import constants
from machina.python.saved_model import loader
from machina.python.saved_model import main_op
from machina.python.saved_model import method_name_updater
from machina.python.saved_model import signature_constants
from machina.python.saved_model import signature_def_utils
from machina.python.saved_model import tag_constants
from machina.python.saved_model import utils
from machina.python.saved_model.fingerprinting import Fingerprint
from machina.python.saved_model.fingerprinting import read_fingerprint
from machina.python.saved_model.load import load
from machina.python.saved_model.save import save
# pylint: enable=unused-import
# pylint: disable=wildcard-import
from machina.python.saved_model.simple_save import *
# pylint: enable=wildcard-import
