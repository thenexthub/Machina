# Tests of TensorFlow image kernels written using the Python API.

load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")
load(
    "//machina/tools/test:performance.bzl",
    "tf_py_benchmark_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "attention_ops_test",
    size = "small",
    srcs = ["attention_ops_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:image_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "decode_bmp_op_test",
    size = "small",
    srcs = ["decode_bmp_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:image_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "decode_compressed_op_test",
    size = "small",
    srcs = ["decode_compressed_op_test.py"],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/platform:client_testlib",
        "@pypi_zstandard//:pkg",
    ],
)

tf_py_strict_test(
    name = "decode_image_op_test",
    size = "small",
    srcs = ["decode_image_op_test.py"],
    data = ["//machina/core:image_testdata"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "decode_jpeg_op_test",
    srcs = ["decode_jpeg_op_test.py"],
    data = ["//machina/core:image_testdata"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
    ],
)

tf_py_strict_test(
    name = "decode_png_op_test",
    size = "small",
    srcs = ["decode_png_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "decode_raw_op_test",
    size = "small",
    srcs = ["decode_raw_op_test.py"],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "draw_bounding_box_op_test",
    size = "small",
    srcs = ["draw_bounding_box_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:image_ops_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

# TODO(gpapan): Revisit the gradient of extract_image_patches_op to resolve
# http://b/31080670.
cuda_py_strict_test(
    name = "extract_image_patches_grad_test",
    size = "medium",
    srcs = ["extract_image_patches_grad_test.py"],
    shard_count = 15,
    tags = [
        "no_oss",  # b/241024908
        "no_rocm",
        "nomac",  # b/181799478
        "notap",  # b/31080670
    ],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/eager/polymorphic_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "extract_image_patches_op_test",
    size = "small",
    srcs = ["extract_image_patches_op_test.py"],
    # TODO(b/144432983): S32 convolutions should not be auto-clustered.
    xla_enable_strict_auto_jit = False,
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "extract_volume_patches_grad_test",
    size = "medium",
    srcs = ["extract_volume_patches_grad_test.py"],
    shard_count = 50,
    tags = [
        "no_gpu",  # b/171837334
        "no_oss",  # Test times out on oss-nightly cpu builds
        "no_pip",
        "nomac",  # b/139946976
        "notap",  # b/31080670
    ],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "extract_volume_patches_op_test",
    size = "small",
    srcs = ["extract_volume_patches_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)
