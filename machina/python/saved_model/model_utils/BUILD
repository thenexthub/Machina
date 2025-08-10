# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Description:
#   Keras saving and loading libraries.

load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:__subpackages__"],
    licenses = ["notice"],
)

py_strict_library(
    name = "model_utils",
    srcs = ["__init__.py"],
    deps = [
        ":export_output",
        ":export_utils",
        ":mode_keys",
    ],
)

py_strict_library(
    name = "export_output",
    srcs = ["export_output.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_util",
        "//machina/python/saved_model:signature_def_utils",
    ],
)

py_strict_test(
    name = "export_output_test",
    srcs = ["export_output_test.py"],
    deps = [
        ":export_output",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:metrics",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:signature_constants",
    ],
)

py_strict_library(
    name = "export_utils",
    srcs = ["export_utils.py"],
    deps = [
        ":export_output",
        ":mode_keys",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:op_selector",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:signature_def_utils",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/saved_model:utils",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//machina/python/util:object_identity",
    ],
)

py_strict_test(
    name = "export_test",
    srcs = ["export_test.py"],
    deps = [
        ":export_output",
        ":export_utils",
        ":mode_keys",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:signature_def_utils",
    ],
)

py_strict_library(
    name = "mode_keys",
    srcs = ["mode_keys.py"],
    deps = ["//machina/python/util:compat"],
)

py_strict_test(
    name = "mode_keys_test",
    srcs = ["mode_keys_test.py"],
    deps = [
        ":mode_keys",
        "//machina/python/platform:client_testlib",
    ],
)
