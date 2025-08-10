load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina/lite:build_def.bzl", "tflite_copts")
load("//machina/lite/tools/evaluation/tasks:build_def.bzl", "task_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

exports_files(
    ["task_executor_c_api.h"],
    visibility = ["//machina/lite/tools/evaluation/tasks:__subpackages__"],
)

cc_library(
    name = "task_executor",
    srcs = ["task_executor.cc"],
    hdrs = ["task_executor.h"],
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//machina/lite/tools:command_line_flags",
        "//machina/lite/tools:logging",
        "//machina/lite/tools/evaluation:evaluation_delegate_provider",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "task_executor_main",
    srcs = ["task_executor_main.cc"],
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":task_executor",
        "//machina/lite/tools:logging",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "task_executor_c_api",
    srcs = ["task_executor_c_api.cc"],
    hdrs = [
        "task_executor_c_api.h",
    ],
    copts = tflite_copts(),
    visibility = [
        "//machina/lite/tools/evaluation/tasks:__subpackages__",
    ],
    deps = [
        ":task_executor",
        "//machina/lite/tools:logging",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
    ],
)
