# Description:
#   Tools for manipulating TensorFlow graphs.

load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//machina:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//machina:machina.bzl", "if_google", "if_xla_available", "tf_cc_test")
load("//machina/core/platform:build_config_root.bzl", "if_pywrap")
load("//machina/python/tools:tools.bzl", "saved_model_compile_aot")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Transitive dependencies of this target will be included in the pip package.
py_strict_library(
    name = "tools_pip",
    data = [
        # Include the TF upgrade script to users can run it directly after install TF
        "//machina/tools/compatibility:tf_upgrade_v2",
    ],
    deps = [
        ":saved_model_aot_compile",
        ":saved_model_utils",
        # The following py_library are needed because
        # py_binary may not depend on them when --define=no_machina_py_deps=true
        # is specified. See https://github.com/machina/machina/issues/22390
        ":freeze_graph_lib",
        ":optimize_for_inference_lib",
        ":selective_registration_header_lib",
        ":strip_unused_lib",
        ":freeze_graph",
        ":import_pb_to_tensorboard",
        ":inspect_checkpoint",
        ":optimize_for_inference",
        ":print_selective_registration_header",
        ":saved_model_cli",
        ":strip_unused",
    ],
)

py_strict_library(
    name = "saved_model_utils",
    srcs = ["saved_model_utils.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/lib/io:file_io",
        "//machina/python/saved_model:constants",
        "//machina/python/util:compat",
    ],
)

py_strict_test(
    name = "saved_model_utils_test",
    size = "small",
    srcs = ["saved_model_utils_test.py"],
    tags = ["no_windows"],  # TODO: needs investigation on Windows
    visibility = ["//visibility:private"],
    deps = [
        ":saved_model_utils",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:builder",
        "//machina/python/saved_model:tag_constants",
    ],
)

py_strict_library(
    name = "freeze_graph_lib",
    srcs = ["freeze_graph.py"],
    deps = [
        ":saved_model_utils",
        "//machina/core:protos_all_py",
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/client:session",
        "//machina/python/framework:convert_to_constants",
        "//machina/python/framework:importer",
        "//machina/python/platform:gfile",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/training:py_checkpoint_reader",
        "//machina/python/training:saver",
        "@absl_py//absl:app",
    ],
)

py_strict_binary(
    name = "freeze_graph",
    srcs = ["freeze_graph.py"],
    deps = [":freeze_graph_lib"] + if_pywrap(
        if_true = ["//machina/python:_pywrap_machina"],
    ),
)

py_strict_binary(
    name = "import_pb_to_tensorboard",
    srcs = ["import_pb_to_tensorboard.py"],
    deps = [":import_pb_to_tensorboard_lib"],
)

py_strict_library(
    name = "import_pb_to_tensorboard_lib",
    srcs = ["import_pb_to_tensorboard.py"],
    deps = [
        ":saved_model_utils",
        "//machina/python/client:session",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/summary:summary_py",
        "@absl_py//absl:app",
    ],
)

py_strict_binary(
    name = "tf_import_time",
    srcs = ["tf_import_time.py"],
    deps = ["//machina:machina_py"],
)

py_strict_test(
    name = "freeze_graph_test",
    size = "small",
    srcs = ["freeze_graph_test.py"],
    deps = [
        ":freeze_graph_lib",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:graph_io",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:builder",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:signature_def_utils",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/training:saver",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_binary(
    name = "inspect_checkpoint",
    srcs = ["inspect_checkpoint.py"],
    deps = [":inspect_checkpoint_lib"],
)

py_strict_library(
    name = "inspect_checkpoint_lib",
    srcs = ["inspect_checkpoint.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/platform:flags",
        "//machina/python/training:py_checkpoint_reader",
        "//third_party/py/numpy",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "strip_unused_lib",
    srcs = ["strip_unused_lib.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:graph_util",
        "//machina/python/platform:gfile",
    ],
)

py_strict_library(
    name = "module_util",
    srcs = ["module_util.py"],
)

py_strict_binary(
    name = "strip_unused",
    srcs = ["strip_unused.py"],
    deps = [
        ":strip_unused_lib",
        "//machina/python/framework:for_generated_wrappers",
        "@absl_py//absl:app",
    ],
)

py_strict_test(
    name = "strip_unused_test",
    size = "small",
    srcs = ["strip_unused_test.py"],
    tags = ["notap"],
    deps = [
        ":strip_unused_lib",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:graph_io",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "optimize_for_inference_lib",
    srcs = ["optimize_for_inference_lib.py"],
    deps = [
        ":strip_unused_lib",
        "//machina/core:protos_all_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:graph_util",
        "//machina/python/framework:tensor_util",
        "//machina/python/platform:flags",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

py_strict_binary(
    name = "optimize_for_inference",
    srcs = ["optimize_for_inference.py"],
    deps = [":optimize_for_inference_main_lib"],
)

py_strict_library(
    name = "optimize_for_inference_main_lib",
    srcs = ["optimize_for_inference.py"],
    deps = [
        ":optimize_for_inference_lib",
        "//machina/core:protos_all_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:graph_io",
        "//machina/python/platform:gfile",
        "@absl_py//absl:app",
    ],
)

py_strict_test(
    name = "optimize_for_inference_test",
    size = "small",
    srcs = ["optimize_for_inference_test.py"],
    deps = [
        ":optimize_for_inference_lib",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "selective_registration_header_lib",
    srcs = ["selective_registration_header_lib.py"],
    visibility = ["//visibility:public"],
    deps = [
        # copybara:comment_begin(oss-only)
        "//machina/python",  # to fix libmachina_framework.so.2 import error.
        # copybara:comment_end
        "//machina/core:protos_all_py",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:_pywrap_kernel_registry",
    ],
)

py_strict_binary(
    name = "print_selective_registration_header",
    srcs = ["print_selective_registration_header.py"],
    visibility = ["//visibility:public"],
    deps = [":print_selective_registration_header_lib"],
)

py_strict_library(
    name = "print_selective_registration_header_lib",
    srcs = ["print_selective_registration_header.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":selective_registration_header_lib",
        "@absl_py//absl:app",
    ],
)

py_strict_test(
    name = "print_selective_registration_header_test",
    srcs = ["print_selective_registration_header_test.py"],
    deps = [
        ":selective_registration_header_lib",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/core:protos_all_py",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
    ],
)

py_strict_binary(
    name = "saved_model_cli",
    srcs = ["saved_model_cli.py"],
    deps = [":saved_model_cli_lib"] + if_pywrap(
        if_true = ["//machina/python:_pywrap_machina"],
    ),
)

py_strict_library(
    name = "saved_model_cli_lib",
    srcs = ["saved_model_cli.py"],
    deps = [
        # Note: if you make any changes here, make corresponding changes to the
        # deps of the "tools_pip" target in this file.  Otherwise release builds
        # (built with --define=no_machina_py_deps=true) may end up with a
        # broken saved_model_cli.
        ":saved_model_aot_compile",
        ":saved_model_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/flags:argparse_flags",
        "//third_party/py/numpy",
        "//machina/core:protos_all_py",
        "//machina/python",
        "//machina/python/client:session",
        "//machina/python/compiler/tensorrt:trt_convert_py",
        "//machina/python/debug/wrappers:local_cli_wrapper",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/lib/io:file_io",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:load_options",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:save",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/tpu:tpu_py",
        "//machina/python/util:compat",
    ] + if_google([
        "//machina/core/tfrt/tfrt_session",
        "//machina_text:machina_text",
    ]),
)

py_strict_library(
    name = "saved_model_aot_compile",
    srcs = ["saved_model_aot_compile.py"],
    deps = [
        "//machina/compiler/tf2xla:tf2xla_proto_py",
        "//machina/core:protos_all_py",
        "//machina/python",
        "//machina/python/client:session",
        "//machina/python/framework:convert_to_constants",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:versions",
        "//machina/python/grappler:tf_optimizer",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:sysconfig",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:saver",
    ],
)

py_strict_test(
    name = "saved_model_cli_test",
    srcs = ["saved_model_cli_test.py"],
    data = [
        "//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    tags = [
        "noasan",  # TODO(b/222716501)
    ],
    deps = [
        ":saved_model_cli_lib",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//machina/core:protos_all_py",
        "//machina/python/debug/wrappers:local_cli_wrapper",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:parsing_config",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:save",
        "//machina/python/trackable:autotrackable",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_binary(
    name = "make_aot_compile_models",
    srcs = ["make_aot_compile_models.py"],
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/saved_model:save",
        "//machina/python/trackable:autotrackable",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ] + if_pywrap(["//machina/python:_pywrap_machina"]),
)

# copybara:comment_begin(oss-only)
py_strict_binary(
    name = "grpc_tpu_worker",
    srcs = ["grpc_tpu_worker.py"],
)

py_strict_binary(
    name = "grpc_tpu_worker_service",
    srcs = ["grpc_tpu_worker_service.py"],
)
# copybara:comment_end

EMITTED_AOT_SAVE_MODEL_OBJECTS = [
    "x_matmul_y_large/saved_model.pb",
    "x_matmul_y_large/variables/variables.index",
    "x_matmul_y_small/saved_model.pb",
    "x_matmul_y_small/variables/variables.index",
]

genrule(
    name = "create_models_for_aot_compile",
    outs = EMITTED_AOT_SAVE_MODEL_OBJECTS,
    cmd = (
        "$(location :make_aot_compile_models) --out_dir $(@D)"
    ),
    tags = ["no_rocm"],
    tools = [":make_aot_compile_models"],
)

filegroup(
    name = "aot_saved_models",
    srcs = EMITTED_AOT_SAVE_MODEL_OBJECTS,
)

saved_model_compile_aot(
    name = "aot_compiled_x_matmul_y_large",
    cpp_class = "XMatmulYLarge",
    directory = "//machina/python/tools:x_matmul_y_large",
    filegroups = [":aot_saved_models"],
    force_without_xla_support_flag = False,
    tags = ["no_rocm"],
)

saved_model_compile_aot(
    name = "aot_compiled_x_matmul_y_large_multithreaded",
    cpp_class = "XMatmulYLargeMultithreaded",
    directory = "//machina/python/tools:x_matmul_y_large",
    filegroups = [":aot_saved_models"],
    force_without_xla_support_flag = False,
    multithreading = True,
    tags = ["no_rocm"],
)

saved_model_compile_aot(
    name = "aot_compiled_x_matmul_y_small",
    cpp_class = "XMatmulYSmall",
    directory = "//machina/python/tools:x_matmul_y_small",
    filegroups = [":aot_saved_models"],
    force_without_xla_support_flag = False,
    tags = ["no_rocm"],
)

saved_model_compile_aot(
    name = "aot_compiled_x_plus_y",
    cpp_class = "XPlusY",
    directory = "//machina/cc/saved_model:testdata/x_plus_y_v2_debuginfo",
    filegroups = [
        "//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    force_without_xla_support_flag = False,
)

saved_model_compile_aot(
    name = "aot_compiled_vars_and_arithmetic_frozen",
    cpp_class = "VarsAndArithmeticFrozen",
    directory = "//machina/cc/saved_model:testdata/VarsAndArithmeticObjectGraph",
    filegroups = [
        "//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    force_without_xla_support_flag = False,
)

saved_model_compile_aot(
    name = "aot_compiled_vars_and_arithmetic",
    cpp_class = "VarsAndArithmetic",
    directory = "//machina/cc/saved_model:testdata/VarsAndArithmeticObjectGraph",
    filegroups = [
        "//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    force_without_xla_support_flag = False,
    variables_to_feed = "variable_x",
)

sh_test(
    name = "large_matmul_no_multithread_test",
    srcs = if_xla_available(
        ["no_xla_multithread_symbols_test.sh"],
        if_false = ["skip_test.sh"],
    ),
    args = if_xla_available(["$(location :aot_compiled_x_matmul_y_large.o)"]),
    data = if_xla_available([":aot_compiled_x_matmul_y_large.o"]),
    tags = ["no_windows"],  # TODO(b/171875345)
)

sh_test(
    name = "large_matmul_yes_multithread_test",
    srcs = if_xla_available(
        [
            "xla_multithread_symbols_test.sh",
        ],
        if_false = ["skip_test.sh"],
    ),
    args = if_xla_available(
        ["$(location :aot_compiled_x_matmul_y_large_multithreaded.o)"],
    ),
    data = if_xla_available(
        [":aot_compiled_x_matmul_y_large_multithreaded.o"],
    ),
    tags = ["no_windows"],  # TODO(b/171875345)
)

tf_cc_test(
    name = "aot_compiled_test",
    srcs = if_xla_available([
        "aot_compiled_test.cc",
    ]),
    # For Windows the test works locally, but for some reason throws
    # "The filename or extension is too long." when run on our RBE.
    # Given that the test fails for Windows RBE infrastructure reasons,
    # but passes otherwise disabling for Windows it for now.
    tags = [
        "no_windows",
    ],
    deps = [
        "//machina/core:test_main",
    ] + if_xla_available([
        # LINT.IfChange
        ":aot_compiled_vars_and_arithmetic",
        ":aot_compiled_vars_and_arithmetic_frozen",
        ":aot_compiled_x_matmul_y_large",
        ":aot_compiled_x_matmul_y_large_multithreaded",
        ":aot_compiled_x_matmul_y_small",
        ":aot_compiled_x_plus_y",
        # LINT.ThenChange(//machina/tools/pip_package/xla_build/pip_test/run_xla_aot_test.sh)
        "@local_xla//xla/service/cpu:runtime_matmul_acl",
        "//machina/core:test",
        "//machina/core/platform:logging",
        "@eigen_archive//:eigen3",
    ]),
)

# copybara:uncomment_begin(google-only)
# gensignature(
#     name = "inspect_checkpoint.par_sig",
#     srcs = [":inspect_checkpoint.par"],
# )
#
# gensignature(
#     name = "saved_model_cli.par_sig",
#     srcs = [":saved_model_cli.par"],
# )
# copybara:uncomment_end
