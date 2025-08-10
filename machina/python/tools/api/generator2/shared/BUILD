load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/python/tools/api/generator2:__subpackages__"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "exported_api",
    srcs = ["exported_api.py"],
)

py_strict_test(
    name = "exported_api_test",
    srcs = ["exported_api_test.py"],
    tags = ["no_pip"],
    deps = [
        ":exported_api",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/python/platform:client_testlib",
    ],
)
