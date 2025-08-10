load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//machina/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load(
    "//machina/lite:special_rules.bzl",
    "tflite_extra_gles_deps",
    "tflite_portable_test_suite_combined",
)
load("//machina/lite/delegates/gpu:build_defs.bzl", "tflite_angle_heapcheck_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "converter",
    srcs = ["converter.cc"],
    hdrs = ["converter.h"],
    deps = [
        "//machina/lite/delegates/gpu:spi",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/gl:command_queue",
        "//machina/lite/delegates/gpu/gl:gl_buffer",
        "//machina/lite/delegates/gpu/gl:gl_program",
        "//machina/lite/delegates/gpu/gl:gl_shader",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_test(
    name = "converter_test",
    size = "small",
    srcs = ["converter_test.cc"],
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":converter",
        ":test_util",
        "//machina/lite/delegates/gpu/common:convert",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/gl:egl_environment",
        "//machina/lite/delegates/gpu/gl:gl_buffer",
        "//machina/lite/delegates/gpu/gl:portable",
        "@com_google_absl//absl/types:span",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "add",
    srcs = ["add.cc"],
    hdrs = ["add.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:convert",
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "add_test",
    srcs = ["add_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":add",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "concat",
    srcs = ["concat.cc"],
    hdrs = ["concat.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "concat_test",
    srcs = ["concat_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":concat",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "conv",
    srcs = ["conv.cc"],
    hdrs = ["conv.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:convert",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:variable",
        "//machina/lite/delegates/gpu/gl/workgroups:ideal_workgroup_picker",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "conv_test",
    srcs = ["conv_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":conv",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "custom_registry",
    srcs = ["custom_registry.cc"],
    hdrs = ["custom_registry.h"],
    deps = [
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "depthwise_conv",
    srcs = ["depthwise_conv.cc"],
    hdrs = ["depthwise_conv.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:convert",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:variable",
        "//machina/lite/delegates/gpu/gl/workgroups:ideal_workgroup_picker",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "depthwise_conv_test",
    srcs = ["depthwise_conv_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":depthwise_conv",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "elementwise",
    srcs = ["elementwise.cc"],
    hdrs = ["elementwise.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:object",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "elementwise_test",
    srcs = ["elementwise_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":elementwise",
        ":test_util",
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:tensor",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "fully_connected",
    srcs = ["fully_connected.cc"],
    hdrs = ["fully_connected.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:convert",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "fully_connected_test",
    srcs = ["fully_connected_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":fully_connected",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "lstm",
    srcs = ["lstm.cc"],
    hdrs = ["lstm.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "lstm_test",
    srcs = ["lstm_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":lstm",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "max_unpooling",
    srcs = ["max_unpooling.cc"],
    hdrs = ["max_unpooling.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "max_unpooling_test",
    srcs = ["max_unpooling_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":max_unpooling",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "mean",
    srcs = ["mean.cc"],
    hdrs = ["mean.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "mean_test",
    srcs = ["mean_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":mean",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "mul",
    srcs = ["mul.cc"],
    hdrs = ["mul.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:convert",
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "mul_test",
    srcs = ["mul_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":mul",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "pad",
    srcs = ["pad.cc"],
    hdrs = ["pad.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "pad_test",
    srcs = ["pad_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":pad",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "pooling",
    srcs = ["pooling.cc"],
    hdrs = ["pooling.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "pooling_test",
    srcs = ["pooling_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":pooling",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "prelu",
    srcs = ["prelu.cc"],
    hdrs = ["prelu.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:convert",
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "prelu_test",
    srcs = ["prelu_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":prelu",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "quantize_and_dequantize",
    srcs = ["quantize_and_dequantize.cc"],
    hdrs = ["quantize_and_dequantize.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "quantize_and_dequantize_test",
    srcs = ["quantize_and_dequantize_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantize_and_dequantize",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/kernels/internal:quantization_util",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "relu",
    srcs = ["relu.cc"],
    hdrs = ["relu.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "relu_test",
    srcs = ["relu_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":relu",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "resampler",
    srcs = ["resampler.cc"],
    hdrs = ["resampler.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "resampler_test",
    srcs = ["resampler_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":resampler",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "reshape",
    srcs = ["reshape.cc"],
    hdrs = ["reshape.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "reshape_test",
    srcs = ["reshape_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":reshape",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "resize",
    srcs = ["resize.cc"],
    hdrs = ["resize.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "resize_test",
    srcs = ["resize_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":resize",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "slice",
    srcs = ["slice.cc"],
    hdrs = ["slice.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "slice_test",
    srcs = ["slice_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":slice",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "softmax",
    srcs = ["softmax.cc"],
    hdrs = ["softmax.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "softmax_test",
    srcs = ["softmax_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":softmax",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "space_to_depth",
    srcs = ["space_to_depth.cc"],
    hdrs = ["space_to_depth.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:any",
    ],
)

cc_test(
    name = "space_to_depth_test",
    srcs = ["space_to_depth_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":space_to_depth",
        ":test_util",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    linkopts = select({
        "//machina:android": [
            "-lEGL",
            "-lGLESv3",
            "-ldl",
            "-lm",
        ],
        # copybara:uncomment_begin(google-only)
        # "//machina/lite/delegates/gpu:tflite_gpu_angle": [],
        # "//machina/lite/delegates/gpu:tflite_gpu_extra_gles_deps": [],
        # copybara:uncomment_end
        "//conditions:default": [
            "-lEGL",
            "-lGLESv3",
        ],
    }),
    deps = [
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/gl:api",
        "//machina/lite/delegates/gpu/gl:compiler_options",
        "//machina/lite/delegates/gpu/gl:egl_environment",
        "//machina/lite/delegates/gpu/gl:gl_buffer",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:object_manager",
        "//machina/lite/delegates/gpu/gl:request_gpu_info",
        "//machina/lite/delegates/gpu/gl:runtime_options",
        "//machina/lite/delegates/gpu/gl/workgroups:default_calculator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest",
    ] + tflite_extra_gles_deps(),
)

cc_library(
    name = "tile",
    srcs = ["tile.cc"],
    hdrs = ["tile.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:any",
    ],
)

cc_test(
    name = "tile_test",
    srcs = ["tile_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":test_util",
        ":tile",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

cc_library(
    name = "transpose_conv",
    srcs = ["transpose_conv.cc"],
    hdrs = ["transpose_conv.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:convert",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "//machina/lite/delegates/gpu/gl:variable",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "transpose_conv_test",
    srcs = ["transpose_conv_test.cc"],
    linkstatic = True,
    tags = tf_gpu_tests_tags() + [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":test_util",
        ":transpose_conv",
        "//machina/lite/delegates/gpu/common:operations",
    ] + tflite_angle_heapcheck_deps(),
)

TFLITE_GPU_BINARY_RELEASE_OPERATORS = [
    "add",
    "concat",
    "conv",
    "depthwise_conv",
    "elementwise",
    "fully_connected",
    "lstm",
    "mul",
    "pad",
    "pooling",
    "prelu",
    "quantize_and_dequantize",
    "relu",
    "mean",
    "resampler",
    "reshape",
    "resize",
    "slice",
    "softmax",
    "space_to_depth",
    "tile",
    "transpose_conv",
]

NON_TFLITE_GPU_BINARY_RELEASE_OPERATORS = [
    "max_unpooling",
]

cc_library(
    name = "registry",
    srcs = ["registry.cc"],
    hdrs = ["registry.h"],
    visibility = ["//visibility:public"],
    deps = [":" + op_name for op_name in TFLITE_GPU_BINARY_RELEASE_OPERATORS] +
           select({
               "//machina/lite/delegates/gpu:tflite_gpu_binary_release": [],
               "//conditions:default": NON_TFLITE_GPU_BINARY_RELEASE_OPERATORS,
           }) + [
        ":custom_registry",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/gl:node_shader",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tflite_portable_test_suite_combined(combine_conditions = {"deps": [":test_util"]})
