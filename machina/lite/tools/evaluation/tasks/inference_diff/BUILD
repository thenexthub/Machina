load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
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

cc_library(
    name = "run_eval_lib",
    srcs = ["run_eval.cc"],
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//machina/lite/core/c:common",
        "//machina/lite/tools:command_line_flags",
        "//machina/lite/tools:logging",
        "//machina/lite/tools/evaluation:evaluation_delegate_provider",
        "//machina/lite/tools/evaluation:evaluation_stage",
        "//machina/lite/tools/evaluation/proto:evaluation_config_cc_proto",
        "//machina/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "//machina/lite/tools/evaluation/stages:inference_profiler_stage",
        "//machina/lite/tools/evaluation/tasks:task_executor",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "run_eval",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":run_eval_lib",
        "//machina/lite/tools/evaluation/tasks:task_executor_main",
    ],
)
