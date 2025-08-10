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

# Description:
#   Keras saving and loading files for SavedModel.

load("//machina:py.default.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/python/keras/saving:__subpackages__",
    ],
    licenses = ["notice"],
)

py_library(
    name = "load_context",
    srcs = ["load_context.py"],
    visibility = ["//machina/python/distribute:__pkg__"],
)

py_library(
    name = "saved_model",
    srcs = [
        "base_serialization.py",
        "constants.py",
        "json_utils.py",
        "layer_serialization.py",
        "load.py",
        "load_context.py",
        "metric_serialization.py",
        "model_serialization.py",
        "network_serialization.py",
        "save.py",
        "save_impl.py",
        "serialized_attributes.py",
        "utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:signature_def_utils",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/saved_model:utils",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//machina/python/util:tf_decorator_py",
    ],
)
