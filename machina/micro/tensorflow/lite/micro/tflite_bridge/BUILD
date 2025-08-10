load(
    "//machina/lite/micro:build_def.bzl",
    "tflm_cc_library",
)

package(
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "flatbuffer_conversions_bridge",
    srcs = [
        "flatbuffer_conversions_bridge.cc",
    ],
    hdrs = [
        "flatbuffer_conversions_bridge.h",
    ],
    visibility = [
        "//machina/lite/micro:__pkg__",
    ],
    deps = [
        ":micro_error_reporter",
        "//machina/lite/c:common",
        "//machina/lite/core/api",
        "//machina/lite/schema:schema_fbs",
    ],
)

tflm_cc_library(
    name = "micro_error_reporter",
    srcs = [
        "micro_error_reporter.cc",
    ],
    hdrs = [
        "micro_error_reporter.h",
    ],
    deps = [
        "//machina/lite/core/api:error_reporter",
        "//machina/lite/micro:micro_compatibility",
        "//machina/lite/micro:micro_log",
    ],
)
