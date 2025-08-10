# Description:
# Scripts used to generate TensorFlow Python API.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//machina/python/tools/api/generator:api_gen.bzl", "MACHINA_API_GEN_PACKAGES")
load("//machina/python/tools/api/generator:api_init_files.bzl", "KERAS_API_INIT_FILES", "MACHINA_API_INIT_FILES")
load("//machina/python/tools/api/generator:api_init_files_v1.bzl", "KERAS_API_INIT_FILES_V1", "MACHINA_API_INIT_FILES_V1")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

exports_files(
    [
        "create_python_api.py",
    ],
)

py_strict_library(
    name = "create_python_api",
    srcs = ["create_python_api.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":doc_srcs",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "doc_srcs",
    srcs = ["doc_srcs.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/util:tf_export",
    ],
)

py_strict_test(
    name = "create_python_api_test",
    srcs = ["create_python_api_test.py"],
    deps = [
        ":create_python_api",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
    ],
)

py_strict_test(
    name = "machina_doc_srcs_test",
    srcs = ["doc_srcs_test.py"],
    args = [
        "--package=machina.python",
        "--api_name=machina",
    ] + KERAS_API_INIT_FILES + KERAS_API_INIT_FILES_V1 + MACHINA_API_INIT_FILES + MACHINA_API_INIT_FILES_V1,
    main = "doc_srcs_test.py",
    deps = [
        ":doc_srcs",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/python:no_contrib",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "output_init_files_test",
    srcs = ["output_init_files_test.py"],
    args = [
        "--packages=" + ",".join(MACHINA_API_GEN_PACKAGES),
    ],
    data = [
        "api_init_files.bzl",
        "api_init_files_v1.bzl",
    ],
    tags = [
        "no_pip",
    ],
    deps = [
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/dtensor/python:dtensor",
        "//machina/lite/python:analyzer",
        "//machina/lite/python:lite",
        "//machina/lite/python/authoring",
        "//machina/python:modules_with_exports",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
        "//machina/python/util:tf_decorator_py",
    ],
)

bzl_library(
    name = "api_init_files",
    srcs = ["api_init_files.bzl"],
    visibility = ["//machina/python/tools/api/generator2:__subpackages__"],
)
