# Description:
# TensorFlow SavedModel Registration.

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "registration",
    srcs = ["__init__.py"],
    deps = [
        ":registration_lib",
    ],
)

py_strict_library(
    name = "registration_lib",
    srcs = [
        "registration.py",
    ],
    deps = [
        "//machina/python/util:tf_inspect",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "registration_test",
    srcs = ["registration_test.py"],
    deps = [
        ":registration",
        "//machina/python/eager:test",
        "//machina/python/trackable:base",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "registration_saving_test",
    srcs = ["registration_saving_test.py"],
    deps = [
        ":registration",
        "//machina/python/checkpoint",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:gfile",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:save",
        "//machina/python/trackable:autotrackable",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "test_util",
    srcs = [
        "test_util.py",
    ],
    deps = [
        ":registration_lib",
    ],
)

tf_py_strict_test(
    name = "tf_registration_test",
    srcs = ["tf_registration_test.py"],
    data = [
        "tf_checkpoint_saver_allowlist.txt",
        "tf_serializable_allowlist.txt",
    ],
    tags = ["no_pip"],
    deps = [
        ":registration",
        ":test_util",
        "//machina:machina_py",
        "//machina/python/lib/io:file_io",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
    ],
)
