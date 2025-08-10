load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.bzl", "py_test")  # @unused
load("//machina:machina.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/grappler:tf_optimizer",
        "//machina/python/training:saver",
    ],
)

cuda_py_strict_test(
    name = "dct_ops_test",
    srcs = ["dct_ops_test.py"],
    shard_count = 16,
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops/signal:dct_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "fft_ops_test",
    size = "medium",
    srcs = ["fft_ops_test.py"],
    shard_count = 2,
    tags = [
        "optonly",
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:spectral_ops_gen",
        "//machina/python/ops/signal:fft_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "mel_ops_test",
    srcs = ["mel_ops_test.py"],
    deps = [
        ":test_util",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops/signal:mel_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "mfcc_ops_test",
    srcs = ["mfcc_ops_test.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops/signal:mfcc_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "reconstruction_ops_test",
    srcs = ["reconstruction_ops_test.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/signal:reconstruction_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "shape_ops_test",
    srcs = ["shape_ops_test.py"],
    deps = [
        ":test_util",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/signal:shape_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "spectral_ops_test",
    size = "large",
    srcs = ["spectral_ops_test.py"],
    tags = [
        "no_rocm",
        "nomac",
    ],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/signal:spectral_ops",
        "//machina/python/ops/signal:window_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "window_ops_test",
    srcs = ["window_ops_test.py"],
    shard_count = 4,
    tags = [
        "no_rocm",
        "no_windows_gpu",
    ],
    deps = [
        ":test_util",
        "//machina/python/eager:context",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops/signal:window_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
