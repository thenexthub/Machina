load("//machina/lite:build_def.bzl", "tflite_copts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "builtin_op_data.h",
        "tflite_types.h",
    ],
    visibility = [
        "//machina/lite:__subpackages__",
    ],
)

# LINT.IfChange(common)
cc_library(
    name = "tflite_common",
    srcs = [],
    hdrs = [
        "builtin_op_data.h",
        "tflite_types.h",
    ],
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    alwayslink = 1,  # Why?? TODO(b/161243354): eliminate this.
)
# LINT.ThenChange(//machina/lite/core/c:common)
