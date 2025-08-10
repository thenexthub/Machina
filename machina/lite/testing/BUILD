load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:strict.default.bzl", "py_strict_binary", "py_strict_library")
load(
    "//machina:machina.bzl",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_features_nolayering_check_if_ios",
)
load("//machina:machina.default.bzl", "pybind_extension")
load("//machina/core/platform:build_config_root.bzl", "if_pywrap")
load(
    "//machina/lite:build_def.bzl",
    "tflite_custom_android_library",
    "tflite_custom_cc_library",
)
load("//machina/lite:special_rules.bzl", "tflite_portable_test_suite")
load(
    "//machina/lite/testing:build_def.bzl",
    "delegate_suffix",
    "gen_zip_test",
    "gen_zipped_test_file",
    "generated_test_models_all",
)
load("//machina/lite/testing:tflite_model_test.bzl", "tflite_model_test")
# copybara:uncomment(oss-unused) load("//tools/build_defs/build_test:build_test.bzl", "build_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

exports_files([
    "build_def.bzl",
    "generated_examples_zip_test.cc",
    "tflite_diff_example_test.cc",
    "init_machina.h",
    "init_machina.cc",
])

_test_size_override = {
    # copybara:comment_begin(oss-only)
    "merged_models_with-flex_6": "large",
    "merged_models_with-flex_9": "large",
    # copybara:comment_end
}

[gen_zip_test(
    name = "zip_test_%s%s" % (
        test_name,
        delegate_suffix(delegate),
    ),
    size = _test_size_override.get(test_name, "medium"),
    srcs = ["generated_examples_zip_test.cc"],
    args = args + select({
        "//machina:android": [],
        "//conditions:default": [
            "--zip_file_path=$(location :zip_%s%s)" % (
                test_name,
                delegate_suffix(delegate),
            ),
            # TODO(angerson) We may be able to add an external unzip binary instead
            # of relying on an existing one for OSS builds.
            #"--unzip_binary_path=/usr/bin/unzip",
        ],
    }),
    conversion_mode = conversion_mode,
    # copybara:uncomment_begin(no special handling for Android in OSS)
    # data = select({
    # "//machina:android": [],
    # "//conditions:default": [
    # ":zip_%s%s" % (
    # test_name,
    # delegate_suffix(delegate),
    # ),
    # "//third_party/unzip",
    # ],
    # }),
    # copybara:uncomment_end_and_comment_begin
    data = [":zip_%s%s" % (
        test_name,
        delegate_suffix(delegate),
    )],
    # copybara:comment_end
    delegate = delegate,
    tags = tags + [
        "gen_zip_test",
        "tflite_not_portable_intentional",
    ],
    test_name = test_name,
    deps = [
        ":parse_testdata_lib",
        ":tflite_driver",
        ":tflite_driver_delegate_providers",
        ":util",
        "//machina/lite:builtin_op_data",
        "//machina/lite:framework",
        "//machina/lite:string",
        "//machina/lite/kernels:builtin_ops",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_googlesource_code_re2//:re2",
    ] + select({
        "//conditions:default": [
            "@local_tsl//tsl/platform:env",
            "@local_tsl//tsl/platform:status",
            "@local_xla//xla/tsl/platform:subprocess",
            "@local_xla//xla/tsl/util:command_line_flags",
        ],
        "//machina:android": [
            "//machina/core:portable_machina_lib",
            "//machina/core:portable_machina_test_lib",
        ],
    }),
) for conversion_mode, delegate, test_name, tags, args in generated_test_models_all()]

test_suite(
    name = "generated_zip_tests",
    tags = [
        "gen_zip_test",
    ],
)

py_strict_library(
    name = "mlir_convert",
    srcs = ["mlir_convert.py"],
    data = [
        "//machina/compiler/mlir/lite:tf_tfl_translate",
    ],
    deps = [
        ":zip_test_utils",
        "//machina/lite/python:lite",
        "//machina/lite/python:test_util",
        "//machina/python/platform:resource_loader",
        "//machina/python/saved_model:signature_constants",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "op_tests",
    srcs = glob(["op_tests/*.py"]),
    deps = [
        ":zip_test_utils",
        "//third_party/py/numpy",
        "//machina:machina_py",
        # copybara:uncomment_begin(b/186563810)
        # "//third_party/py/machina_addons",
        # copybara:uncomment_end
        "//machina/python/framework:test_lib",
        "//machina/lite/python:lite",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:rnn",
    ],
)

py_strict_library(
    name = "generate_examples_lib",
    srcs = ["generate_examples_lib.py"],
    deps = [
        ":op_tests",
        ":zip_test_utils",
        "//machina:machina_py",
    ],
)

py_strict_library(
    name = "zip_test_utils",
    srcs = ["zip_test_utils.py"],
    deps = [
        ":generate_examples_report",
        "//machina:machina_py",
        "//machina/lite/python:lite",
        "//machina/lite/tools:flatbuffer_utils",
        "//machina/python/framework:convert_to_constants",
        "//machina/python/saved_model:signature_constants",
        "//third_party/py/numpy",
        "@ml_dtypes_py//ml_dtypes",
    ] + if_pywrap(
        if_false = [
            ":_pywrap_string_util",
        ],
        if_true = [
            "//machina/lite/python:pywrap_tflite",
        ],
    ),
)

py_strict_binary(
    name = "generate_examples",
    srcs = ["generate_examples.py"],
    deps = [
        ":generate_examples_lib",
        ":mlir_convert",
        "//machina:machina_py",
    ] + if_pywrap(
        if_true = ["//machina/python:_pywrap_machina"],
    ),
)

py_strict_library(
    name = "generate_examples_report",
    srcs = ["generate_examples_report.py"],
)

cc_library(
    name = "parse_testdata_lib",
    srcs = ["parse_testdata.cc"],
    hdrs = ["parse_testdata.h"],
    deps = [
        ":message",
        ":split",
        ":test_runner",
        "//machina/lite:framework",
        "//machina/lite:string",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core:framework_stable",
    ],
)

cc_library(
    name = "matchers",
    testonly = True,
    srcs = ["matchers.h"],
    hdrs = ["matchers.h"],
    deps = [
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "matchers_test",
    srcs = ["matchers_test.cc"],
    deps = [
        ":matchers",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "message",
    srcs = ["message.cc"],
    hdrs = ["message.h"],
    deps = [":tokenize"],
)

cc_test(
    name = "message_test",
    srcs = ["message_test.cc"],
    deps = [
        ":message",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "split",
    srcs = ["split.cc"],
    hdrs = ["split.h"],
    deps = [
        "//machina/lite:string",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "split_test",
    size = "small",
    srcs = ["split_test.cc"],
    deps = [
        ":split",
        "//machina/lite:string",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "join",
    hdrs = ["join.h"],
    deps = ["//machina/lite:string"],
)

cc_test(
    name = "join_test",
    size = "small",
    srcs = ["join_test.cc"],
    deps = [
        ":join",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tflite_driver",
    srcs = ["tflite_driver.cc"],
    hdrs = ["tflite_driver.h"],
    deps = [
        ":join",
        ":result_expectations",
        ":split",
        ":test_runner",
        "//machina/lite:builtin_op_data",
        "//machina/lite:framework",
        "//machina/lite:string",
        "//machina/lite:string_util",
        "//machina/lite/core:framework",
        "//machina/lite/core/api:op_resolver",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/kernels:custom_ops",
        "//machina/lite/kernels:reference_ops",
        "//machina/lite/kernels:test_delegate_providers_lib",
        "//machina/lite/kernels/gradient:gradient_ops",
        "//machina/lite/kernels/parse_example",
        "//machina/lite/kernels/perception:perception_ops",
        "//machina/lite/tools:logging",
        "//machina/lite/tools/delegates:delegate_provider_hdr",
        "//machina/lite/tools/evaluation:utils",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ] + select({
        "//machina:ios": [],
        "//conditions:default": ["//machina/lite/delegates/flex:delegate"],
    }),
)

# A convenient library of tflite delegate execution providers for tests based
# on the `tflite_driver` library.
cc_library(
    name = "tflite_driver_delegate_providers",
    deps = [
        "//machina/lite/tools/delegates:coreml_delegate_provider",
        "//machina/lite/tools/delegates:default_execution_provider",
        "//machina/lite/tools/delegates:external_delegate_provider",
        "//machina/lite/tools/delegates:gpu_delegate_provider",
        "//machina/lite/tools/delegates:hexagon_delegate_provider",
        "//machina/lite/tools/delegates:nnapi_delegate_provider",
        "//machina/lite/tools/delegates:xnnpack_delegate_provider",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "tflite_driver_test",
    size = "small",
    srcs = ["tflite_driver_test.cc"],
    data = [
        "//machina/lite:testdata/add_quantized_int8.bin",
        "//machina/lite:testdata/multi_add.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":test_runner",
        ":tflite_driver",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tokenize",
    srcs = ["tokenize.cc"],
    hdrs = ["tokenize.h"],
    deps = [
        "//machina/lite:string",
    ],
)

cc_test(
    name = "tokenize_test",
    srcs = ["tokenize_test.cc"],
    deps = [
        ":tokenize",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_runner",
    hdrs = ["test_runner.h"],
    deps = [
        "//machina/lite:string",
    ],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "//machina/lite:string",
        "//machina/lite/core/api",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:flags",
    ],
)

cc_test(
    name = "test_runner_test",
    srcs = ["test_runner_test.cc"],
    deps = [
        ":test_runner",
        "//machina/lite:string",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_binary(
    name = "nnapi_example",
    srcs = ["nnapi_example.cc"],
    deps = [
        ":parse_testdata_lib",
        ":tflite_driver",
    ],
)

cc_library(
    name = "tf_driver",
    srcs = ["tf_driver.cc"],
    hdrs = ["tf_driver.h"],
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        ":join",
        ":split",
        ":test_runner",
        "//machina/core:protos_all_cc",
        "//machina/lite:string",
        "//machina/lite:string_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ] + select({
        "//conditions:default": [
            "//machina/core:core_cpu",
            "//machina/core:framework",
            "//machina/core:lib",
            "//machina/core:machina",
        ],
        "//machina:android": [
            "//machina/core:portable_machina_lib",
        ],
        "//machina:ios": [
            "//machina/core:portable_machina_lib",
        ],
    }),
)

tf_cc_test(
    name = "tf_driver_test",
    size = "small",
    srcs = ["tf_driver_test.cc"],
    data = ["//machina/lite:testdata/multi_add.pb"],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":tf_driver",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/lite:string",
        "//machina/lite:string_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "generate_testspec",
    srcs = ["generate_testspec.cc"],
    hdrs = ["generate_testspec.h"],
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        ":join",
        ":split",
        ":test_runner",
        ":tf_driver",
        ":tflite_driver",
        "//machina/core:protos_all_cc",
        "//machina/lite:string",
        "@com_google_absl//absl/log:check",
    ] + select({
        "//conditions:default": [
            "//machina/core:framework",
        ],
        "//machina:android": [
            "//machina/core:portable_machina_lib",
        ],
        "//machina:ios": [
            "//machina/core:portable_machina_lib",
        ],
    }),
)

tf_cc_test(
    name = "generate_testspec_test",
    size = "small",
    srcs = ["generate_testspec_test.cc"],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":generate_testspec",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "init_machina",
    srcs = [
        "init_machina.cc",
    ],
    hdrs = [
        "init_machina.h",
    ],
    visibility = [
        # copybara:uncomment_begin(internal brella benchmark)
        # "//learning/brain/mobile/lite/brella_benchmark:__subpackages__",
        # copybara:uncomment_end
        "//machina/lite/delegates/flex:__subpackages__",
        "//machina/lite/tools/benchmark:__subpackages__",
    ],
    deps = select({
        "//conditions:default": [
            "//machina/core:lib",
        ],
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//machina:ios": [
            "//machina/core:portable_machina_lib_lite",
        ],
    }),
)

cc_library(
    name = "tflite_diff_util",
    srcs = ["tflite_diff_util.cc"],
    hdrs = ["tflite_diff_util.h"],
    deps = [
        ":generate_testspec",
        ":parse_testdata_lib",
        ":test_runner",
        ":tflite_driver",
        "//machina/lite:framework",
        "//machina/lite:string",
    ],
)

cc_library(
    name = "tflite_diff_flags",
    hdrs = ["tflite_diff_flags.h"],
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        ":split",
        ":tflite_diff_util",
        ":tflite_driver",
        "@com_google_absl//absl/strings",
    ] + select({
        "//conditions:default": [
            "//machina/core:framework_internal",
            "//machina/core:lib",
        ],
        "//machina:android": [
            "//machina/core:portable_machina_lib",
        ],
        "//machina:ios": [
            "//machina/core:portable_machina_lib",
        ],
    }),
)

tf_cc_test(
    name = "tflite_diff_example_test",
    size = "medium",
    srcs = ["tflite_diff_example_test.cc"],
    args = [
        "--machina_model=third_party/machina/lite/testdata/multi_add.pb",
        "--tflite_model=third_party/machina/lite/testdata/multi_add.bin",
        "--input_layer=a,b,c,d",
        "--input_layer_type=float,float,float,float",
        "--input_layer_shape=1,3,4,3:1,3,4,3:1,3,4,3:1,3,4,3",
        "--output_layer=x,y",
    ],
    data = [
        "//machina/lite:testdata/multi_add.bin",
        "//machina/lite:testdata/multi_add.pb",
    ],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_oss",  # needs test data
        "tflite_not_portable",
    ],
    deps = [
        ":init_machina",
        ":tflite_diff_flags",
        ":tflite_diff_util",
    ],
)

tf_cc_binary(
    name = "tflite_diff",
    srcs = ["tflite_diff_example_test.cc"],
    deps = [
        ":init_machina",
        ":tflite_diff_flags",
        ":tflite_diff_util",
    ],
)

tflite_model_test(
    name = "tflite_model_example_test",
    input_layer = "a,b,c,d",
    input_layer_shape = "1,8,8,3:1,8,8,3:1,8,8,3:1,8,8,3",
    input_layer_type = "float,float,float,float",
    output_layer = "x,y",
    tags = [
        "no_cuda_on_cpu_tap",
        "no_oss",  # needs test data
        "tflite_not_portable",  # TODO(b/134772701): Enable after making this a proper GTest.
    ],
    machina_model_file = "//machina/lite:testdata/multi_add.pb",
)

cc_library(
    name = "string_util_lib",
    srcs = ["string_util.cc"],
    hdrs = ["string_util.h"],
    deps = [
        "//machina/lite:string",
        "//machina/lite:string_util",
        "//machina/lite/python/interpreter_wrapper:numpy",
        "//machina/lite/python/interpreter_wrapper:python_utils",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",
    ],
)

cc_library(
    name = "result_expectations",
    srcs = ["result_expectations.cc"],
    hdrs = ["result_expectations.h"],
    deps = [
        ":split",
        "//machina/lite:framework",
        "//machina/lite:string_util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
)

# A selective built tflite for testing.
tflite_custom_cc_library(
    name = "test_tflite_lib",
    models = [
        "//machina/lite:testdata/add.bin",
        "//machina/lite:testdata/lstm.bin",
    ],
)

cc_test(
    name = "selective_build_test",
    srcs = ["selective_build_test.cc"],
    data = [
        "//machina/lite:testdata/add.bin",
        "//machina/lite:testdata/lstm.bin",
    ],
    tags = [
        "no_mac",  # b/161990368
        "tflite_not_portable",
    ],
    deps = [
        ":test_tflite_lib",
        "//machina/core:tflite_portable_logging",
        "//machina/lite:framework",
        "//machina/lite/core:cc_api_stable",
        "//machina/lite/core/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

pybind_extension(
    name = "_pywrap_string_util",
    srcs = [
        "string_util_wrapper.cc",
    ],
    hdrs = ["string_util.h"],
    additional_stubgen_deps = [
        "//third_party/py/numpy:numpy",
    ],
    common_lib_packages = [
        "litert/python",
        "machina/lite/python",
    ],
    enable_stub_generation = True,
    features = ["-use_header_modules"],
    pytype_srcs = [
        "_pywrap_string_util.pyi",
    ],
    wrap_py_init = True,
    deps = [
        ":string_util_lib",
        "//machina/lite/python/interpreter_wrapper:numpy",
        "//machina/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tflite_portable_test_suite()

tflite_custom_android_library(
    name = "customized_tflite_for_add_ops",
    experimental = True,
    models = ["//machina/lite:testdata/add.bin"],
    visibility = ["//visibility:public"],
)

edgetpu_ops = [
    "add",
    "avg_pool",
    "concat",
    "conv",  # high error
    "conv_relu",
    "conv_relu1",
    "conv_relu6",
    "depthwiseconv",  # high error
    "expand_dims",
    "fully_connected",
    "l2norm",  # high error
    "maximum",
    "max_pool",
    "mean",
    "minimum",
    "mul",
    "pad",  # high error
    "pack",
    "relu",
    "relu1",
    "relu6",
    "reshape",
    "resize_bilinear",
    "resize_nearest_neighbor",
    "sigmoid",
    "slice",
    "softmax",
    "space_to_depth",
    "split",
    "squeeze",
    "strided_slice",
    "sub",
    "sum",  # high error
    "tanh",
    "transpose",
    "transpose_conv",
]

# copybara:uncomment_begin(google-only)
# [gen_zipped_test_file(
#     name = "zip_%s_edgetpu" % op_name,
#     file = "%s_edgetpu.zip" % op_name,
#     flags = " --make_edgetpu_tests",
# ) for op_name in edgetpu_ops]
#
# edgetpu_targets = [":zip_%s_edgetpu" % op_name for op_name in edgetpu_ops]
#
# build_test(
#     name = "gen_edgetpu_tests",
#     targets = edgetpu_targets,
# )
# copybara:uncomment_end
