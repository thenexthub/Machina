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
#   Contains the Keras Mixed Precision API (TensorFlow version).

load("//machina:py.default.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        # TODO(scottzhu): Remove these two deps and convert the test to integration test.
        "//machina/python/distribute:__pkg__",  # For collective_all_reduce_strategy_test
        "//machina/python/keras:__subpackages__",
        "//machina/tools/pip_package:__pkg__",
        "//machina/tools/pip_package:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//machina/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "mixed_precision_experimental",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":get_layer_policy",
        ":loss_scale_optimizer",
        ":policy",
    ],
)

py_library(
    name = "policy",
    srcs = [
        "policy.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":device_compatibility_check",
        "//machina/python/framework:dtypes",
        "//machina/python/platform:tf_logging",
        "//machina/python/training/experimental:mixed_precision_global_state",
    ],
)

py_library(
    name = "device_compatibility_check",
    srcs = ["device_compatibility_check.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/framework:config",
    ],
)

py_library(
    name = "get_layer_policy",
    srcs = ["get_layer_policy.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/keras/engine:base_layer",
    ],
)

py_library(
    name = "autocast_variable",
    srcs = [
        "autocast_variable.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//machina/python/distribute:distribute_utils",
        "//machina/python/distribute:ps_values",
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_conversion_registry",
        "//machina/python/framework:tensor_shape",
        "//machina/python/keras/distribute",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/types:core",
    ],
)

py_library(
    name = "loss_scale",
    srcs = ["loss_scale.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/keras/utils:generic_utils",
        "//machina/python/training/experimental:loss_scale",
    ],
)

py_library(
    name = "loss_scale_optimizer",
    srcs = ["loss_scale_optimizer.py"],
    srcs_version = "PY3",
    deps = [
        ":loss_scale",
        "//machina/python/distribute:collective_all_reduce_strategy",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:mirrored_strategy",
        "//machina/python/distribute:one_device_strategy",
        "//machina/python/distribute:tpu_strategy",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/keras/optimizer_v2",
        "//machina/python/ops:cond",
        "//machina/python/ops:variable_v1",
        "//machina/python/trackable:base_delegate",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/keras",
        "//machina/python/ops:cond",
    ],
)
