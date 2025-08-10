#include "third_party/absl/strings/str_cat.h"
#Description:
# TensorFlow SavedModel.

load("//machina:strict.default.bzl", "py_strict_binary")
load(
    "//machina:machina.bzl",
    "if_android",
    "if_google",
    "if_mobile",
    "if_not_mobile",
    "if_not_windows_or_mac",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "filegroup")
load(
    "//machina/core/platform:build_config_root.bzl",
    "if_static",
    "if_static_and_not_mobile",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")
load(
    "//machina/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "loader.h",
    "testdata/chunked_saved_model/chunked_model/saved_model.cpb",
    "testdata/chunked_saved_model/chunked_model/saved_model.pbtxt",
])

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    deps = ["@com_google_absl//absl/strings"],
)

cc_library(
    name = "signature_constants",
    hdrs = ["signature_constants.h"],
)

cc_library(
    name = "tag_constants",
    hdrs = ["tag_constants.h"],
)

# copybara:uncomment_begin(google-only)
# cc_library(
#     name = "mobile_only_deps",
#     visibility = ["//visibility:private"],
#     deps = if_mobile(["//machina/core:portable_machina_lib"]),
# )
# copybara:uncomment_end

cc_library(
    name = "reader",
    srcs = ["reader.cc"],
    hdrs = ["reader.h"],
    deps = [
        ":constants",
        ":metrics",
        ":util",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status:statusor",
    ] + if_google([
        "//machina/tools/proto_splitter:merge",
    ]) + if_not_mobile([
        # TODO(b/111634734): :lib and :protos_all contain dependencies that
        # cannot be built on mobile platforms. Instead, include the appropriate
        # tf_lib depending on the build platform.
        "@com_google_absl//absl/memory:memory",
        "//machina/core:lib",
        "//machina/core/util/tensor_bundle:byteswaptensor",
    ]),
)

tf_cc_test(
    name = "reader_test",
    srcs = ["reader_test.cc"],
    data = [
        ":saved_model_test_files",
    ],
    linkstatic = 1,
    deps = [
        ":constants",
        ":metrics",
        ":reader",
        ":tag_constants",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core/platform:resource_loader",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "loader",
    hdrs = ["loader.h"],
    deps = [
        ":loader_lite",
    ] + if_static_and_not_mobile([
        "//machina/core:direct_session",
        "//machina/core:all_kernels",
    ] + if_google(
        ["//machina/core/platform/default/build_config:machina_platform_specific"],
        [],
    )) + if_not_mobile([
        "//machina/core:core_cpu",
        "//machina/core:lib",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
    ]) + if_android([
        "//machina/core:portable_machina_lib",
    ]),
)

cc_library(
    name = "loader_lite",
    hdrs = ["loader.h"],
    deps = if_static([
        ":loader_lite_impl",
    ]) + if_not_mobile([
        "//machina/core:core_cpu",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ]),
)

cc_library(
    name = "loader_lite_impl",
    srcs = ["loader.cc"],
    hdrs = ["loader.h"],
    deps = [
        ":constants",
        ":fingerprinting",
        ":loader_util",
        ":reader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ] + if_not_mobile([
        ":metrics",
        ":util",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/util/tensor_bundle:naming",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "bundle_v2",
    srcs = ["bundle_v2.cc"],
    hdrs = ["bundle_v2.h"],
    deps = [
        ":constants",
        ":fingerprinting",
        ":metrics",
        ":reader",
        ":util",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:strcat",
        "//machina/core/platform:tstring",
        "//machina/core/util/tensor_bundle",
        "//machina/core/util/tensor_bundle:byteswaptensor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:strcat",
    ],
)

cc_library(
    name = "loader_util",
    srcs = ["loader_util.cc"],
    hdrs = ["loader_util.h"],
    deps = [":constants"] + if_not_mobile([
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
    ]),
)

tf_cc_test(
    name = "bundle_v2_test",
    srcs = ["bundle_v2_test.cc"],
    data = [
        ":saved_model_test_files",
    ],
    linkstatic = 1,
    deps = [
        ":bundle_v2",
        ":metrics",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:test",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "saved_model_bundle_test",
    srcs = ["saved_model_bundle_test.cc"],
    data = [
        ":saved_model_test_files",
    ],
    linkstatic = 1,
    deps = [
        ":constants",
        ":loader",
        ":metrics",
        ":reader",
        ":signature_constants",
        ":tag_constants",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

tf_cc_test(
    name = "saved_model_bundle_lite_test",
    srcs = ["saved_model_bundle_lite_test.cc"],
    data = [
        ":saved_model_test_files",
    ],
    linkstatic = 1,
    deps = [
        ":constants",
        ":loader",
        ":signature_constants",
        ":tag_constants",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

# A subset of the TF2 saved models can be generated with this tool.
py_strict_binary(
    name = "testdata/generate_saved_models",
    srcs = ["testdata/generate_saved_models.py"],
    data = [
        ":saved_model_asset_data",
        ":saved_model_static_hashtable_asset_data",
    ],
    deps = [
        "//machina/python/client:session",
        "//machina/python/compat:v2_compat",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/module",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model",
        "//machina/python/saved_model:save_options",
        "//machina/python/trackable:asset",
        "@absl_py//absl:app",
    ],
)

# copybara:uncomment_begin(google-only)
#
# py_strict_binary(
#     name = "testdata/generate_chunked_models",
#     srcs = ["testdata/generate_chunked_models.py"],
#     deps = [
#         "//machina/python/compat:v2_compat",
#         "//machina/python/eager:def_function",
#         "//machina/python/framework:constant_op",
#         "//machina/python/lib/io:file_io",
#         "//machina/python/module",
#         "//machina/python/saved_model:loader",
#         "//machina/python/saved_model:save",
#         "//machina/python/saved_model:save_options",
#         "//machina/python/util:compat",
#         "//machina/tools/proto_splitter:constants",
#         "//machina/tools/proto_splitter/python:saved_model",
#         "//third_party/py/numpy",
#         "@absl_py//absl:app",
#         "@absl_py//absl/flags",
#     ],
# )
#
# copybara:uncomment_end

# TODO(b/32673259): add a test to continuously validate these files.
filegroup(
    name = "saved_model_test_files",
    srcs = glob([
        "testdata/AssetModule/**",
        "testdata/half_plus_two_pbtxt/**",
        "testdata/half_plus_two_main_op/**",
        "testdata/half_plus_two/**",
        "testdata/half_plus_two_v2/**",
        "testdata/x_plus_y_v2_debuginfo/**",
        "testdata/CyclicModule/**",
        "testdata/StaticHashTableModule/**",
        "testdata/VarsAndArithmeticObjectGraph/**",
        "testdata/fuzz_generated/**",
        "testdata/SimpleV1Model/**",
        "testdata/OptimizerSlotVariableModule/**",
        "testdata/chunked_saved_model/**",
    ]),
)

filegroup(
    name = "saved_model_fingerprinting_test_files",
    srcs = glob([
        "testdata/bert2/**",
        "testdata/bert1/**",
    ]),
)

alias(
    name = "saved_model_half_plus_two",
    actual = ":saved_model_test_files",
)

filegroup(
    name = "saved_model_asset_data",
    srcs = [
        "testdata/test_asset.txt",
    ],
)

filegroup(
    name = "saved_model_static_hashtable_asset_data",
    srcs = [
        "testdata/static_hashtable_asset.txt",
    ],
)

exports_files(
    glob([
        "testdata/half_plus_two_pbtxt/**",
        "testdata/half_plus_two_main_op/**",
        "testdata/half_plus_two/**",
        "testdata/half_plus_two_v2/**",
        "testdata/x_plus_y_v2_debuginfo/**",
        "testdata/CyclicModule/**",
        "testdata/VarsAndArithmeticObjectGraph/**",
        "testdata/fuzz_generated/**",
    ]),
)

# Linked directly into ":machina_framework".
cc_library(
    name = "metrics_impl",
    srcs = [
        "metrics.cc",
        "metrics.h",
    ],
    visibility = [
        "//machina:__pkg__",
        "//machina/python:__pkg__",
    ],
    deps = [
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@jsoncpp_git//:jsoncpp",
    ] + if_not_mobile([
        "//machina/core:lib",
    ]) + if_android([
        "//machina/core:portable_machina_lib_lite",
    ]),
    alwayslink = True,
)

cc_library(
    name = "metrics",
    hdrs = ["metrics.h"],
    visibility = [
        "//machina/cc/saved_model/image_format:__subpackages__",
        "//machina/python/saved_model:__subpackages__",
    ],
    deps = if_static([
        ":metrics_impl",
    ]) + if_not_mobile([
        "//machina/core:lib",
    ]) + if_android([
        "//machina/core:portable_machina_lib_lite",
    ]) + [
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "metrics_test",
    size = "small",
    srcs = ["metrics_test.cc"],
    deps = [
        ":metrics",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@jsoncpp_git//:jsoncpp",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core/framework:tensor_proto_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/protobuf:for_core_protos_cc",
    ] + if_not_mobile(["//machina/core:lib"]) + if_android(["//machina/core:portable_machina_lib_lite"]),
)

cc_library(
    name = "test_utils",
    testonly = True,
    hdrs = ["test_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:test",
        "//machina/core/platform:protobuf",
    ],
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    deps = [
        ":test_utils",
        ":util",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:tensor_shape_proto_cc",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/protobuf:for_core_protos_cc",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)

# Linked directly into ":machina_framework".
cc_library(
    name = "fingerprinting_impl",
    srcs = [
        "fingerprinting.cc",
        "fingerprinting.h",
    ],
    visibility = [
        "//machina:__pkg__",
        "//machina/python:__pkg__",
    ],
    deps = [
        ":constants",
        ":fingerprinting_x_platform_utils",
        "//machina/core:protos_all_cc",
        "//machina/core/graph/regularization:simple_delete",
        "//machina/core/graph/regularization:util",
        "//machina/core/util/tensor_bundle:naming",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:types",
    ] + if_not_mobile([
        "//machina/core:lib",
    ]) + if_android([
        "//machina/core:portable_machina_lib_lite",
    ]) + if_not_windows_or_mac([
        ":fingerprinting_utils",
        "//machina/tools/proto_splitter/cc:util",
    ]),
    alwayslink = True,
)

cc_library(
    name = "fingerprinting",
    hdrs = ["fingerprinting.h"],
    visibility = [
        "//learning/brain/contrib/hub/server/ingestion:__subpackages__",
        "//learning/brain/contrib/tpu_modeling:__subpackages__",
        "//learning/metadata/artifactoid/cc:__subpackages__",
        "//learning/tfx/pipeline/util:__subpackages__",
        "//machina/core/tfrt:__subpackages__",
        "//machina/python/saved_model:__subpackages__",
    ],
    deps = if_static([
        ":fingerprinting_impl",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/status:statusor",
        "//machina/core:protos_all_cc",
    ]) + if_not_mobile([
        "//machina/core:lib",
    ]) + if_android([
        "//machina/core:portable_machina_lib_lite",
    ]),
)

cc_library(
    name = "fingerprinting_utils",
    srcs = ["fingerprinting_utils.cc"],
    hdrs = ["fingerprinting_utils.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":constants",
        ":fingerprinting_x_platform_utils",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/util/tensor_bundle:naming",
        "//machina/tools/proto_splitter:chunk_proto_cc",
        "//machina/tools/proto_splitter:merge",
        "//machina/tools/proto_splitter/cc:util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/records:record_reader",
    ],
    alwayslink = True,
)

cc_library(
    name = "fingerprinting_x_platform_utils",
    srcs = ["fingerprinting_x_platform_utils.cc"],
    hdrs = ["fingerprinting_x_platform_utils.h"],
    deps = [
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:random",
    ],
)

tf_cc_test(
    name = "fingerprinting_utils_test",
    srcs = ["fingerprinting_utils_test.cc"],
    data = [
        "//machina/tools/proto_splitter/testdata:many-field.cpb",
        "//machina/tools/proto_splitter/testdata:split-standard.cpb",
    ],
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":fingerprinting_utils",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:path",
        "//machina/core/platform:protobuf",
        "//machina/tools/proto_splitter:chunk_proto_cc",
        "//machina/tools/proto_splitter/cc:util",
        "//machina/tools/proto_splitter/testdata:test_message_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

tf_cc_test(
    name = "fingerprinting_chunked_test",
    size = "small",
    srcs = ["fingerprinting_chunked_test.cc"],
    data = [
        ":saved_model_fingerprinting_test_files",
        ":saved_model_test_files",
    ],
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":fingerprinting",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core/platform:path",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)

# copybara:uncomment_end

tf_cc_test(
    name = "fingerprinting_test",
    size = "small",
    srcs = ["fingerprinting_test.cc"],
    data = [
        ":saved_model_fingerprinting_test_files",
        ":saved_model_test_files",
    ],
    deps = [
        ":fingerprinting",
        "//machina/cc:cc_ops",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tf_cc_fuzz_test(
    name = "saved_model_fuzz",
    srcs = ["saved_model_fuzz.cc"],
    deps = [
        ":constants",
        ":loader",
        ":tag_constants",
        "//machina/core:core_cpu",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
    ],
)
