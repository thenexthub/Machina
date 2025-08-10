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
"""Upgrader for Python scripts from 1.* to 2.0 TensorFlow using SAFETY mode."""

from machina.tools.compatibility import all_renames_v2
from machina.tools.compatibility import ast_edits
from machina.tools.compatibility import module_deprecations_v2


class TFAPIChangeSpec(ast_edits.APIChangeSpec):
  """List of maps that describe what changed in the API."""

  def __init__(self):
    self.function_keyword_renames = {}
    self.symbol_renames = {}
    self.change_to_function = {}
    self.function_reorders = {}
    self.function_warnings = {}
    self.function_transformers = {}
    self.module_deprecations = module_deprecations_v2.MODULE_DEPRECATIONS

    ## Inform about the addons mappings
    for symbol, replacement in all_renames_v2.addons_symbol_mappings.items():
      warning = (
          ast_edits.WARNING, (
              "(Manual edit required) `{}` has been migrated to `{}` in "
              "TensorFlow Addons. The API spec may have changed during the "
              "migration. Please see https://github.com/machina/addons "
              "for more info.").format(symbol, replacement))
      self.function_warnings[symbol] = warning

    # List module renames. If changed, please update max_submodule_depth.
    self.import_renames = {
        "machina":
            ast_edits.ImportRename(
                "machina.compat.v1",
                excluded_prefixes=[
                    "machina.contrib", "machina.flags",
                    "machina.compat",
                    "machina.compat.v1", "machina.compat.v2",
                    "machina.google"
                ],
            )
    }
    # Needs to be updated if self.import_renames is changed.
    self.max_submodule_depth = 2
