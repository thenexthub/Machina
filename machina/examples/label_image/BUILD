# Description:
#   TensorFlow C++ inference example for labeling images.

load("//machina:strict.default.bzl", "py_strict_binary")
load("//machina:machina.bzl", "tf_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

exports_files(["data/grace_hopper.jpg"])

tf_cc_binary(
    name = "label_image",
    srcs = [
        "main.cc",
    ],
    linkopts = select({
        "//machina:android": [
            "-pie",
            "-landroid",
            "-ljnigraphics",
            "-llog",
            "-lm",
            "-z defs",
            "-s",
            "-Wl,--exclude-libs,ALL",
        ],
        "//conditions:default": ["-lm"],
    }),
    deps = select({
        "//machina:android": [
            "//machina/core:portable_machina_lib",
            # cc:android_machina_image_op is for including jpeg/gif/png
            # decoder to enable real-image evaluation on Android
            "//machina/core/kernels/image:android_machina_image_op",
        ],
        "//conditions:default": [
            "//machina/core:core_cpu",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:lib",
            "//machina/core:protos_all_cc",
            "//machina/core:machina",
        ],
    }) + [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "@local_xla//xla/tsl/util:command_line_flags",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:types",
        # cc:cc_ops is used to include image ops (for label_image)
        # Jpg, gif, and png related code won't be included
        "//machina/cc:cc_ops",
    ],
)

py_strict_binary(
    name = "label_image_py",
    srcs = ["label_image.py"],
    main = "label_image.py",
    deps = [
        "//machina:machina_py",
        "//third_party/py/numpy",
    ],
)
