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
###############################################################################=

load("//machina:machina.bzl", "py_test")
load("//machina:machina.default.bzl", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler/mlir/lite/integrations:__subpackages__",
        "//third_party/odml/litert/litert/python/tools/model_utils:__subpackages__",
    ],
    licenses = ["notice"],
)

pybind_extension(
    name = "model_utils_core_pybind",
    srcs = [
        "model_utils_core_pybind.cc",
    ],
    deps = [
        "//machina/compiler/mlir:init_mlir",
        "//machina/compiler/mlir/lite:flatbuffer_export",
        "//machina/compiler/mlir/lite:flatbuffer_import",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/quantization/ir:QuantOps",
        "//machina/compiler/mlir/lite/stablehlo:prepare_hlo",
        "//machina/compiler/mlir/machina:convert_tensor",
        "//machina/python/lib/core:ndarray_tensor",
        "//machina/python/lib/core:py_func_lib",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:CAPIIRHeaders",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:FuncTransforms",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MLIRBindingsPythonHeaders",
        "@toolchain-project//mlir:MLIRBindingsPythonHeadersAndDeps",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:vhlo_ops",
    ],
)

py_test(
    name = "py_bindings_test",
    srcs = ["py_bindings_test.py"],
    deps = [
        "//machina/compiler/mlir/lite/integrations/python/mlir",
    ],
)
