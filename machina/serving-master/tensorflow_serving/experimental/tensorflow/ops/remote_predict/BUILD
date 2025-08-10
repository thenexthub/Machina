# Description:
# Remote Predict Op Prototype

# buildifier: disable=same-origin-load
# Placeholder: load py_test
load("@org_machina//machina:machina.bzl", "tf_custom_op_library", "tf_gen_op_libs", "tf_gen_op_wrapper_cc", "tf_gen_op_wrapper_py", "tf_kernel_library")
load("@org_machina//machina:machina.bzl", "tf_custom_op_py_library")
# buildifier: enable=same-origin-load

licenses(["notice"])

cc_library(
    name = "remote_predict_op_kernel_lib",
    srcs = [
        "kernels/prediction_service_grpc.cc",
        "kernels/remote_predict_op_kernel.cc",
    ],
    hdrs = [
        "kernels/prediction_service_grpc.h",
        "kernels/remote_predict_op_kernel.h",
    ],
    deps = [
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/apis:predict_cc_proto",
        "//machina_serving/apis:prediction_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf_lite",
        "@org_machina//machina/core:framework_headers_lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/kernels:ops_util_hdrs",
        "@org_machina//machina/core/platform:statusor",
    ],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["remote_predict_op"],
    deps = [
        "@org_machina//machina/core:lib",
    ],
)

tf_gen_op_wrapper_cc(
    name = "gen_cc_remote_predict_op",
    out_ops_file = "cc/ops/remote_predict_op",
    deps = [":remote_predict_op_op_lib"],
)

cc_library(
    name = "remote_predict_ops_cc",
    srcs = ["cc/ops/remote_predict_op.cc"],
    hdrs = ["cc/ops/remote_predict_op.h"],
    deps = [
        ":remote_predict_op_op_lib",
        "@org_machina//machina/cc:const_op",
        "@org_machina//machina/cc:ops",
        "@org_machina//machina/cc:scope",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_remote_predict_op",
    out = "ops/gen_remote_predict_op.py",
    deps = [":remote_predict_op_op_lib"],
)

tf_kernel_library(
    name = "remote_predict_op_kernel",
    srcs = [
        "kernels/prediction_service_grpc.cc",
        "kernels/remote_predict_op_kernel.cc",
    ],
    hdrs = [
        "kernels/prediction_service_grpc.h",
        "kernels/remote_predict_op_kernel.h",
    ],
    deps = [
        ":remote_predict_op_op_lib",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/apis:predict_cc_proto",
        "//machina_serving/apis:prediction_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:cc_wkt_protos",
        "@com_google_protobuf//:protobuf_lite",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core/kernels:ops_util",
        "@org_machina//machina/core/kernels:split_lib",
        "@org_machina//machina/core/platform:statusor",
    ],
)

tf_custom_op_library(
    name = "python/ops/_remote_predict_op.so",
    srcs = [
        "ops/remote_predict_op.cc",
    ],
    deps = [":remote_predict_op_kernel_lib"],
)

tf_custom_op_py_library(
    name = "remote_predict_py",
    srcs = glob(["python/ops/*.py"]) + ["__init__.py"],
    dso = [":python/ops/_remote_predict_op.so"],
    kernels = [
        ":remote_predict_op_kernel",
        ":remote_predict_op_op_lib",
    ],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
    ],
    visibility = ["//machina_serving/experimental/example:__subpackages__"],
    deps = [
        ":gen_remote_predict_op",
        "@org_machina//machina/python/framework:for_generated_wrappers",
    ],
)

cc_test(
    name = "remote_predict_ops_cc_test",
    size = "small",
    srcs = [
        "kernels/remote_predict_op_kernel.h",
        "kernels/remote_predict_op_kernel_test.cc",
    ],
    deps = [
        ":remote_predict_ops_cc",
        "//machina_serving/apis:model_cc_proto",
        "//machina_serving/apis:predict_cc_proto",
        "//machina_serving/apis:prediction_service_cc_proto",
        "//machina_serving/core/test_util:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:cc_wkt_protos",
        "@com_google_protobuf//:protobuf_lite",
        "@org_machina//machina/cc:client_session",
        "@org_machina//machina/cc:const_op",
        "@org_machina//machina/core:all_kernels",
        "@org_machina//machina/core:direct_session",
        "@org_machina//machina/core:framework",
        "@org_machina//machina/core:lib",
        "@org_machina//machina/core:protos_all_cc",
        "@org_machina//machina/core:test",
        "@org_machina//machina/core:testlib",
        "@org_machina//machina/core/kernels:ops_util",
    ],
)

cc_library(
    name = "prediction_service_grpc",
    srcs = ["kernels/prediction_service_grpc.cc"],
    hdrs = ["kernels/prediction_service_grpc.h"],
    deps = [
        "//machina_serving/apis:prediction_service_cc_proto",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@org_machina//machina/core/platform:statusor",
    ],
)

cc_test(
    name = "prediction_service_grpc_test",
    size = "small",
    srcs = [
        "kernels/prediction_service_grpc_test.cc",
    ],
    deps = [
        ":prediction_service_grpc",
        "//machina_serving/core/test_util:test_main",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/time",
        "@org_machina//machina/core:testlib",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//visibility:private"],
)
