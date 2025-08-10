load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "v2_compat",
    srcs = ["v2_compat.py"],
    visibility = [
        "//smartass/brain/ops:__subpackages__",
        "//machina:internal",
    ],
    deps = [
        "//machina/python:tf2",
        "//machina/python/eager:monitoring",
        "//machina/python/framework:ops",
        "//machina/python/framework:registry",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:control_flow_v2_toggles",
        "//machina/python/ops:resource_variables_toggle",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "compat",
    srcs = ["compat.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python/platform:tf_logging",
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "compat_test",
    size = "small",
    srcs = ["compat_test.py"],
    tags = ["nofwdcompat"],
    deps = [
        ":compat",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "disable_v2_behavior_test",
    size = "small",
    srcs = ["disable_v2_behavior_test.py"],
    deps = [
        ":v2_compat",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/platform:_pywrap_tf2",
        "//machina/python/platform:client_testlib",
    ],
)
