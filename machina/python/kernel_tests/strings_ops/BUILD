# Tests of TensorFlow math kernels written using the Python API.

load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "as_string_op_test",
    size = "small",
    srcs = ["as_string_op_test.py"],
    tags = ["no_windows"],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "base64_ops_test",
    size = "small",
    srcs = ["base64_ops_test.py"],
    tags = ["nomac"],  # b/35468214
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "reduce_join_op_test",
    size = "small",
    srcs = ["reduce_join_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "regex_full_match_op_test",
    size = "small",
    srcs = ["regex_full_match_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:string_ops_gen",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "regex_replace_op_test",
    size = "small",
    srcs = ["regex_replace_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:string_ops_gen",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "string_bytes_split_op_test",
    size = "small",
    srcs = ["string_bytes_split_op_test.py"],
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_string_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "string_format_op_test",
    size = "small",
    srcs = ["string_format_op_test.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "string_join_op_test",
    size = "small",
    srcs = ["string_join_op_test.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "string_length_op_test",
    size = "small",
    srcs = ["string_length_op_test.py"],
    deps = [
        "//machina/python/framework:test_lib",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "string_lower_op_test",
    size = "small",
    srcs = ["string_lower_op_test.py"],
    deps = [
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "string_split_op_test",
    size = "small",
    srcs = ["string_split_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_string_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "string_strip_op_test",
    size = "small",
    srcs = ["string_strip_op_test.py"],
    deps = [
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "string_to_hash_bucket_op_test",
    size = "small",
    srcs = ["string_to_hash_bucket_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "string_to_number_op_test",
    size = "small",
    srcs = ["string_to_number_op_test.py"],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "string_upper_op_test",
    size = "small",
    srcs = ["string_upper_op_test.py"],
    deps = [
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "substr_op_test",
    size = "small",
    srcs = ["substr_op_test.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "unicode_decode_op_test",
    size = "small",
    srcs = ["unicode_decode_op_test.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:string_ops_gen",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_string_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "unicode_encode_op_test",
    size = "small",
    srcs = ["unicode_encode_op_test.py"],
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_string_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/ops/ragged:ragged_tensor_value",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "unicode_script_op_test",
    size = "small",
    srcs = ["unicode_script_op_test.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "unicode_transcode_op_test",
    size = "small",
    srcs = ["unicode_transcode_op_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "unsorted_segment_join_op_test",
    size = "small",
    srcs = ["unsorted_segment_join_op_test.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:string_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
