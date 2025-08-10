load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "losses",
    srcs = ["losses.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":losses_impl",
        ":util",
    ],
)

py_strict_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:confusion_matrix",
        "//machina/python/ops:math_ops",
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "losses_impl",
    srcs = ["losses_impl.py"],
    deps = [
        ":util",
        "//machina/python/eager:context",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:confusion_matrix",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:weights_broadcast_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

py_strict_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.py"],
    deps = [
        ":util",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)
