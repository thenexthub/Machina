# Description:
#   Wrap NVIDIA TensorRT (http://developer.nvidia.com/tensorrt) with machina
#   and provide TensorRT operators and converter package.
#   APIs are meant to change over time.

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(
    glob(["*test.py"]),
)

py_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
)

py_strict_library(
    name = "tf_trt_integration_test_base_srcs",
    srcs = ["tf_trt_integration_test_base.py"],
    visibility = ["//machina/python/compiler/tensorrt:__pkg__"],
    deps = [
        "//machina/compiler/tf2tensorrt:_pywrap_py_utils",
        "//machina/core:protos_all_py",
        "//machina/python/compiler/tensorrt:trt_convert_py",
        "//machina/python/compiler/tensorrt:utils",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:graph_io",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/profiler:trace",
        "//machina/python/saved_model:builder",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:save",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:signature_def_utils",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/saved_model:utils",
        "//machina/python/tools:saved_model_utils",
        "//machina/python/trackable:autotrackable",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "trt_convert_test_data",
    srcs = [
        "testdata/tf_readvariableop_saved_model/saved_model.pb",
        "testdata/tf_readvariableop_saved_model/variables/variables.data-00000-of-00001",
        "testdata/tf_readvariableop_saved_model/variables/variables.index",
        "testdata/tf_variablev2_saved_model/saved_model.pb",
        "testdata/tf_variablev2_saved_model/variables/variables.data-00000-of-00001",
        "testdata/tf_variablev2_saved_model/variables/variables.index",
        "testdata/tftrt_2.0_saved_model/saved_model.pb",
        "testdata/tftrt_2.0_saved_model/variables/variables.data-00000-of-00002",
        "testdata/tftrt_2.0_saved_model/variables/variables.data-00001-of-00002",
        "testdata/tftrt_2.0_saved_model/variables/variables.index",
    ],
    visibility = ["//machina/python/compiler/tensorrt:__pkg__"],
)

base_tags = [
    "no_cuda_on_cpu_tap",
    "no_rocm",
    "no_windows",
    "nomac",
    # TODO(b/303453873): Re-enable tests once TensorRT has been updated
    "notap",
]

cuda_py_strict_test(
    name = "base_test",
    srcs = ["base_test.py"],
    tags = base_tags + [
        "no_oss",  # TODO(b/165611343): Need to address the failures for CUDA 11 in OSS build.
        "notap",  # TODO(b/291653036): Some of the base tests are failing w/ TensorRT-8
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/compiler/tensorrt:utils",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "conv2d_test",
    srcs = ["conv2d_test.py"],
    tags = base_tags + ["no_oss"],  # b/198501457
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "combined_nms_test",
    srcs = ["combined_nms_test.py"],
    tags = base_tags + ["no_oss"],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/compiler/tensorrt:utils",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:image_ops_impl",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "tf_function_test",
    srcs = ["tf_function_test.py"],
    tags = base_tags + [
        "manual",  # TODO(b/231239602): re-enable once naming issue is resolved.
        "no_oss",  # TODO(b/231239602): re-enable once naming issue is resolved.
        "notap",  # TODO(b/231239602): re-enable once naming issue is resolved.
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/core:protos_all_py",
        "//machina/python/compiler/tensorrt:trt_convert_py",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:constants",
        "//machina/python/saved_model:loader",
        "//machina/python/util:compat",
    ],
)

cuda_py_strict_test(
    name = "batch_matmul_test",
    srcs = ["batch_matmul_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "biasadd_matmul_test",
    srcs = ["biasadd_matmul_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "binary_tensor_weight_broadcast_test",
    srcs = ["binary_tensor_weight_broadcast_test.py"],
    # TODO(b/297490791): Reenable after TensorRT regression has been fixed
    tags = base_tags + ["no_oss"],
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

cuda_py_strict_test(
    name = "bool_test",
    srcs = ["bool_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/compiler/tensorrt:utils",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "cast_test",
    srcs = ["cast_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "concatenation_test",
    srcs = ["concatenation_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "const_broadcast_test",
    srcs = ["const_broadcast_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:nn",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "data_dependent_shape_test",
    srcs = ["data_dependent_shape_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "dynamic_input_shapes_test",
    srcs = ["dynamic_input_shapes_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "identity_output_test",
    srcs = ["identity_output_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "int32_test",
    srcs = ["int32_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "lru_cache_test",
    srcs = ["lru_cache_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "memory_alignment_test",
    srcs = ["memory_alignment_test.py"],
    tags = base_tags + ["no_oss"],  # TODO(b/285588022)
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "multi_connection_neighbor_engine_test",
    srcs = ["multi_connection_neighbor_engine_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "neighboring_engine_test",
    srcs = ["neighboring_engine_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "quantization_test",
    srcs = ["quantization_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "rank_two_test",
    srcs = ["rank_two_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "reshape_transpose_test",
    srcs = ["reshape_transpose_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "topk_test",
    srcs = ["topk_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "trt_engine_op_shape_test",
    srcs = ["trt_engine_op_shape_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:tag_constants",
    ],
)

cuda_py_strict_test(
    name = "trt_mode_test",
    srcs = ["trt_mode_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "unary_test",
    srcs = ["unary_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "vgg_block_nchw_test",
    srcs = ["vgg_block_nchw_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:nn_impl",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "vgg_block_test",
    srcs = ["vgg_block_test.py"],
    tags = base_tags,
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_trt_integration_test_base_srcs",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:nn_impl",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
    ],
)

test_suite(
    name = "tf_trt_integration_test",
    tests = [
        "batch_matmul_test",
        "biasadd_matmul_test",
        "binary_tensor_weight_broadcast_test",
        "bool_test",
        "cast_test",
        "combined_nms_test",
        "concatenation_test",
        "const_broadcast_test",
        "data_dependent_shape_test",
        "dynamic_input_shapes_test",
        "identity_output_test",
        "int32_test",
        "lru_cache_test",
        "memory_alignment_test",
        "multi_connection_neighbor_engine_test",
        "neighboring_engine_test",
        "quantization_test",
        "rank_two_test",
        "reshape_transpose_test",
        "tf_function_test",
        "topk_test",
        "trt_engine_op_shape_test",
        "trt_mode_test",
        "unary_test",
        "vgg_block_nchw_test",
        "vgg_block_test",
    ],
)

test_suite(
    name = "tf_trt_integration_test_no_oss",
    tests = [
        "base_test",
        "conv2d_test",
    ],
)
