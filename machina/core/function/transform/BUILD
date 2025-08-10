load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/core/function/runtime_client:__subpackages__",
    ],
)

licenses(["notice"])

pytype_strict_library(
    name = "transform",
    srcs = [
        "transform.py",
    ],
    visibility = ["//smartass/brain/configure:__subpackages__"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/core/framework:function_proto_py",
        "//machina/core/function/capture:restore_captures",
        "//machina/core/function/runtime_client:runtime_client_py",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:function_def_to_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:custom_gradient",
        "//machina/python/ops:default_gradient",
        "//machina/python/ops:handle_data_util",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
    ],
)

py_strict_test(
    name = "transform_test",
    srcs = ["transform_test.py"],
    tags = ["no_oss"],  # TODO(b/219089812)
    deps = [
        ":transform",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//machina/core/function/testing:test_pass_py",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/module",
        "//machina/python/ops:custom_gradient",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:save",
    ],
)
