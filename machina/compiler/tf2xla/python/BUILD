load("//machina:machina.default.bzl", "tf_custom_op_py_library")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_py_clif_cc",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

tf_py_clif_cc(
    name = "xla_op_registry",
    srcs = ["xla_op_registry.clif"],
    pyclif_deps = [
        "//machina/core/framework:kernel_def_pyclif",
    ],
    deps = [
        "//machina/compiler/tf2xla:xla_compiler",
    ],
)

tf_custom_op_py_library(
    name = "xla",
    srcs = ["xla.py"],
    dso = ["//machina/compiler/tf2xla/ops:_xla_ops.so"],
    kernels = [
        "//machina/compiler/tf2xla/ops:xla_ops",
    ],
    deps = [
        "//machina/compiler/tf2xla/ops:gen_xla_ops",
        "//machina/python/ops:random_ops_util",
        "//machina/python/ops/numpy_ops:np_utils",
        "@local_xla//xla:xla_data_proto_py",
    ],
)
