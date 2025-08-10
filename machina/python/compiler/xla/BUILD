load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_library(
    name = "compiler_py",
    srcs = [
        "__init__.py",
        "jit.py",
    ],
    deps = [
        ":xla",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "jit_test",
    size = "small",
    srcs = ["tests/jit_test.py"],
    tags = [
        "no_windows",  # TODO(b/171385770)
    ],
    xla_enabled = True,
    deps = [
        ":compiler_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:function",
        "//machina/python/framework:op_def_registry",
        "//machina/python/framework:ops",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:gradients",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "xla",
    srcs = ["xla.py"],
    deps = [
        "//machina/compiler/jit:xla_ops_py",
        "//machina/compiler/jit/ops:xla_ops_grad",
        "//machina/core:protos_all_py",
        # Do not remove: required to run xla ops on Cloud.
        "//machina/compiler/tf2xla/python:xla",  # build_cleaner: keep
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/distribute:summary_op_util",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
    ],
)

cuda_py_strict_test(
    name = "xla_test",
    srcs = ["tests/xla_test.py"],
    tags = [
        "no_mac",
        "no_windows",
    ],
    xla_enabled = True,
    deps = [
        ":xla",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:control_flow_util",
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//machina/python/summary:summary_py",
        "//machina/python/tpu:tpu_feed",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "jit_compile_test",
    srcs = ["tests/jit_compile_test.py"],
    tags = [
        "no_mac",
        "no_windows",
    ],
    xla_enabled = True,
    deps = [
        "//machina/python/client:session",
        "//machina/python/eager:backprop",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "pjrt_compile_test",
    srcs = ["tests/pjrt_compile_test.py"],
    env = {
        "TF_MACHINA_MACHINA_XLA_FLAGS": "--tf_xla_use_device_api --tf_xla_enable_xla_devices --tf_xla_enable_device_api_for_gpu",
    },
    tags = [
        "config-cuda-only",
        "gpu",
        "no_oss",
        "requires-gpu-nvidia",
        "xla",
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
    ],
)

tf_py_strict_test(
    name = "pjrt_autoclustering_test",
    srcs = ["tests/pjrt_autoclustering_test.py"],
    env = {
        "TF_MACHINA_MACHINA_XLA_FLAGS": "--tf_xla_use_device_api --tf_xla_enable_xla_devices --tf_xla_enable_device_api_for_gpu --tf_xla_auto_jit=2 --tf_xla_enable_lazy_compilation=false --tf_xla_min_cluster_size=0",
    },
    tags = [
        "config-cuda-only",
        "gpu",
        "no_oss",
        "requires-gpu-nvidia",
        "xla",
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:math_ops",
    ],
)

tf_py_strict_test(
    name = "pjrt_compile_virtual_device_test",
    srcs = ["pjrt_compile_virtual_device_test.py"],
    env = {
        "TF_MACHINA_MACHINA_XLA_FLAGS": "--tf_xla_use_device_api --tf_xla_enable_xla_devices --tf_xla_enable_device_api_for_gpu",
    },
    tags = [
        "config-cuda-only",
        "gpu",
        "no_oss",
        "requires-gpu-nvidia",
        "xla",
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
    ],
)
