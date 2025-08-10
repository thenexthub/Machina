load("//machina/lite:build_def.bzl", "tflite_copts")
load(
    "//machina/lite/micro:build_def.bzl",
    "tflm_cc_library",
    "tflm_copts",
)

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "api",
    srcs = [
        "flatbuffer_conversions.cc",
        "tensor_utils.cc",
    ],
    hdrs = [
        "error_reporter.h",
        "flatbuffer_conversions.h",
        "tensor_utils.h",
    ],
    copts = tflm_copts() + tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":error_reporter",
        "//machina/lite/c:common",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
    ],
)

# We define separate targets for "error_reporter", even though those headers are
# also exported by the "api" target, so that targets which only want to depend
# on these small abstract base class modules can express more fine-grained
# dependencies without pulling in tensor_utils and flatbuffer_conversions.
tflm_cc_library(
    name = "error_reporter",
    hdrs = [
        "error_reporter.h",
        "//machina/compiler/mlir/lite/core/api:error_reporter.h",
    ],
    copts = tflm_copts() + tflite_copts(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//machina/compiler/mlir/lite/core/api:error_reporter",
    ],
)
