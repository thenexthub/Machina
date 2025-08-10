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

load("//machina:machina.default.bzl", "pybind_extension")
load("//machina/compiler/mlir/lite:symlink_files.bzl", "symlink_files")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/compiler/mlir/lite:__subpackages__"],
    features = [
        # Cannot use header_modules (parse_headers feature fails).
        "-use_header_modules",
    ],
    licenses = ["notice"],
)

# These flags are needed for parse_headers feature.
COPTS = [
    "-fexceptions",
    "-frtti",
]

pybind_extension(
    name = "_mlir",
    srcs = [
        "@toolchain-project//mlir:lib/Bindings/Python/MainModule.cpp",
    ],
    copts = COPTS,
    pytype_srcs = [
        ":_mlirPyi",
    ],
    deps = [
        "@toolchain-project//mlir:MLIRBindingsPythonCore",
        "@toolchain-project//mlir:MLIRBindingsPythonNanobindHeaders",
        "@nanobind",
    ],
)

symlink_files(
    name = "_mlirPyi",
    srcs = [
        "@toolchain-project//mlir/python:IRPyIFiles",
        "@toolchain-project//mlir/python:PassManagerPyIFiles",
    ],
    dst = "_mlir",
    flatten = True,
)
