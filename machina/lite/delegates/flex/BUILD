load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "if_android",
    "if_mobile",
    "if_not_mobile",
    "tf_cc_test",
    "tf_features_nolayering_check_if_ios",
    "tf_opts_nortti_if_lite_protos",
    "tf_opts_nortti_if_mobile",
)
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/lite:build_def.bzl", "tflite_copts")
load("//machina/lite:special_rules.bzl", "internal_visibility_allowlist")
load("//machina/lite/delegates/flex:build_def.bzl", "tflite_flex_cc_library", "tflite_flex_shared_library")

default_visibility = [
    "//machina/compiler/mlir/lite:__subpackages__",
    "//machina/lite/android:__subpackages__",
    "//machina/lite/toco/tflite:__subpackages__",
]

#
# This is a TF Lite delegate that is powered by TensorFlow's Eager.
#
package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = default_visibility,
    licenses = ["notice"],
)

exports_files([
    "delegate.h",
    "exported_symbols.lds",
    "version_script.lds",
])

cc_library(
    name = "buffer_map",
    srcs = ["buffer_map.cc"],
    hdrs = ["buffer_map.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_opts_nortti_if_lite_protos(),
    features = tf_features_nolayering_check_if_ios() + if_android(["-layering_check"]),
    deps = [
        ":buffer_map_util",
        ":util",
        "//machina/lite:string",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels/internal:compatibility",
    ] + if_mobile([
        "//machina/core:portable_machina_lib_lite",
    ]) + if_not_mobile([
        "//machina/c:c_api_internal",
        "//machina/core:framework",
    ]),
)

cc_library(
    name = "buffer_map_util",
    srcs = ["buffer_map_util.cc"],
    hdrs = ["buffer_map_util.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_opts_nortti_if_lite_protos(),
    features = tf_features_nolayering_check_if_ios() + if_android(["-layering_check"]),
    deps = [
        ":util",
        "//machina/lite:string_util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/experimental/resource",
    ] + if_mobile([
        "//machina/core:portable_machina_lib_lite",
    ]) + if_not_mobile([
        "//machina/c:c_api_internal",
        "//machina/core:framework",
        "//machina/c:tf_tensor_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:status",
    ]),
)

tf_cc_test(
    name = "buffer_map_test",
    size = "small",
    srcs = ["buffer_map_test.cc"],
    deps = [
        ":buffer_map",
        ":buffer_map_util",
        "//machina/core:framework",
        "//machina/lite:framework",
        "//machina/lite:string_util",
        "//machina/lite:util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/testing:util",
        "@com_google_googletest//:gtest_main",
    ],
)

# Define the standard flex delegate library, that pulls in the standard set
# of TensorFlow ops and kernels, using tflite_flex_cc_library with no
# models parameter. Custom flex delegate can be defined with
# tflite_flex_cc_library if the parameter models is provided. Tensorflow
# user-provided ops could also be supported by passing to additional_deps.
# Ex:
# tflite_flex_cc_library(
#   name = "sample_delegate",
#   models = ["model1.tflite", "model2.tflite"],
#   additional_deps = ["your_custom_ops_lib"],
# )
tflite_flex_cc_library(
    name = "delegate",
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

# Compared to the library above, this one doesn't define a strong symbol for
# AcquireFlexDelegate(). This is useful if one doesn't want the default flex
# delegate to be automatically applied when building the interpreter.
tflite_flex_cc_library(
    name = "delegate_without_symbol",
    link_symbol = False,
    visibility = ["//visibility:public"],
)

# Shared lib target for convenience, pulls in the standard set of TensorFlow
# ops and kernels. The output library name is platform dependent:
#   - Linux/Android: `libmachinalite_flex.so`
#   - Mac: `libmachinalite_flex.dylib`
#   - Windows: `machinalite_flex.dll`
tflite_flex_shared_library(
    name = "machinalite_flex",
)

cc_library(
    name = "delegate_symbol",
    srcs = [
        "delegate_symbol.cc",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_only_runtime",
        "//machina/lite/core/c:c_api_types",
    ],
    alwayslink = 1,
)

# Delegate implementation that does *not* pull in the standard set of TensorFlow
# ops and kernels.
cc_library(
    name = "delegate_only_runtime",
    srcs = [
        "delegate.cc",
        "kernel.cc",
        "kernel.h",
    ],
    hdrs = [
        "delegate.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tf_opts_nortti_if_mobile(),
    features = tf_features_nolayering_check_if_ios() + if_android(["-layering_check"]),
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_map",
        ":delegate_data",
        ":tflite_subgraph_execute",
        ":util",
        "//machina/core:session_options",
        "//machina/core/tfrt/fallback:op_kernel_runner",
        "//machina/lite:kernel_api",
        "//machina/lite:macros",
        "//machina/lite:minimal_logging",
        "//machina/lite:string",
        "//machina/lite:string_util",
        "//machina/lite:util",
        "//machina/lite/core:subgraph",
        "//machina/lite/core/api",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates/utils:simple_delegate",
        "//machina/lite/kernels:kernel_util",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ] + if_mobile([
        "//machina/core:portable_machina_lib_lite",
    ]) + if_not_mobile([
        "//machina/core/common_runtime/eager:context",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:framework",
    ]),
    alwayslink = 1,
)

tf_cc_test(
    name = "delegate_test",
    size = "small",
    srcs = ["delegate_test.cc"],
    tags = [
        "no_gpu",  # GPU + flex is not officially supported.
    ],
    deps = [
        ":delegate",
        ":test_util",
        "//machina/lite:shared_library",
        "//machina/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "delegate_data",
    srcs = ["delegate_data.cc"],
    hdrs = ["delegate_data.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_opts_nortti_if_mobile(),
    features = tf_features_nolayering_check_if_ios() + if_android(["-layering_check"]),
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_map",
        ":subgraph_resource",
        ":util",
        "//machina/lite:cc_api_experimental",
        "//machina/lite:util",
        "//machina/lite/core/c:common",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ] + if_mobile([
        "//machina/core:portable_machina_lib_lite",
    ]) + if_not_mobile([
        "//machina/core/common_runtime/eager:context",
        "//machina/core/common_runtime/eager:core_no_xla",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ]),
)

tf_cc_test(
    name = "delegate_data_test",
    size = "small",
    srcs = ["delegate_data_test.cc"],
    deps = [
        ":delegate_data",
        "//machina/core:test",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/platform:mutex",
        "//machina/core/platform:protobuf",
        "//machina/core/platform:status",
        "//machina/lite:framework",
        "//machina/lite/core/api:error_reporter",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:subgraph_test_util",
        "//machina/lite/testing:util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "subgraph_resource",
    hdrs = ["subgraph_resource.h"],
    compatible_with = get_compatible_with_portable(),
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        "//machina/core/platform:mutex",
        "//machina/core/platform:thread_annotations",
        "//machina/lite:cc_api_experimental",
        "//machina/lite/core/c:common",
    ] + if_mobile([
        "//machina/core:portable_machina_lib_lite",
    ]) + if_not_mobile([
        "//machina/core:framework",
        "//machina/core:lib",
    ]),
)

tf_cc_test(
    name = "kernel_test",
    size = "small",
    srcs = [
        "kernel.h",
        "kernel_test.cc",
    ],
    tags = ["no_gpu"],  # GPU + flex is not officially supported.
    deps = [
        ":delegate",
        ":delegate_data",
        ":test_util",
        "//machina/core/platform:status",
        "//machina/core/tfrt/fallback:op_kernel_runner",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates/utils:simple_delegate",
        "//machina/lite/kernels:kernel_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    visibility = internal_visibility_allowlist(),
    deps = [
        "//machina/c:c_api_internal",
        "//machina/lite:string",
        "//machina/lite/kernels:test_util",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    compatible_with = get_compatible_with_portable(),
    features = tf_features_nolayering_check_if_ios() + if_android(["-layering_check"]),
    #TODO(b/206038955): Consider restrict the visibility to '//third_party/fcp/client:__subpackages__'.
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:protos_all_cc",
        "//machina/lite:kernel_api",
        "//machina/lite:string_util",
        "//machina/lite:util",
        "//machina/lite/c:common",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels/internal:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ] + if_mobile([
        "//machina/core:portable_machina_lib_lite",
    ]) + if_not_mobile([
        "//machina/c:c_api_internal",
        "//machina/core:lib",
        "//machina/core:framework",
        "//machina/core/protobuf:error_codes_proto_impl_cc",
    ]),
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "//machina/c:tf_datatype",
        "//machina/core:framework",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/protobuf:error_codes_proto_impl_cc",
        "//machina/lite:string",
        "//machina/lite:string_util",
        "//machina/lite:util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "allowlisted_flex_ops_test",
    size = "small",
    srcs = [
        "allowlisted_flex_ops_test.cc",
    ],
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        ":delegate",
        "//machina/compiler/mlir/lite/delegates/flex:allowlisted_flex_ops_lib",
        "@com_google_googletest//:gtest_main",
    ] + if_mobile([
        "//machina/core:portable_machina_lib_lite",
    ]) + if_not_mobile([
        "//machina/core:framework",
    ]),
)

# Alias to support selective build of image ops.
# TODO(b/163285312): Remove after machina/core refactoring completed.
cc_library(
    name = "portable_images_lib",
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:portable_gif_internal",
        "//machina/core:portable_jpeg_internal",
        "//machina/core/lib/png:png_io",
        "//machina/core/lib/webp:webp_io",
    ],
)

cc_library(
    name = "tflite_subgraph_execute",
    srcs = ["tflite_subgraph_execute.cc"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_opts_nortti_if_mobile(),
    features = tf_features_nolayering_check_if_ios() + if_android(["-layering_check"]),
    deps = [
        ":buffer_map_util",
        ":subgraph_resource",
        ":util",
        "//machina/lite:cc_api_experimental",
        "//machina/lite:string_util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels/internal:tensor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ] + if_mobile([
        "//machina/core:portable_machina_lib_lite",
    ]) + if_not_mobile([
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/c:tf_tensor_internal",
    ]),
    alwayslink = 1,
)
