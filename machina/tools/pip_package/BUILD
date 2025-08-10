# Description:
#  Tools for building the TensorFlow pip package.

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_syslibs//:build_defs.bzl", "if_not_system_lib")
load(
    "@local_xla//third_party/py:py_import.bzl",
    "py_import",
)
load(
    "@local_xla//third_party/py:py_manylinux_compliance_test.bzl",
    "verify_manylinux_compliance_test",
)
load("@local_xla//third_party/py:python_wheel.bzl", "collect_data_files", "transitive_py_deps")
load("@local_xla//xla/tsl/mkl:build_defs.bzl", "if_enable_mkl", "if_mkl", "if_mkl_ml")
load("//machina:machina.bzl", "if_wheel_dependency", "if_with_tpu_support", "transitive_hdrs")
load("//machina:tf_version.bzl", "TF_SEMANTIC_VERSION_SUFFIX", "TF_VERSION")
load(
    "//machina/core/platform:build_config_root.bzl",
    "if_pywrap",
    "tf_additional_license_deps",
    "tf_cuda_tests_tags",
    "tf_exec_properties",
)
load("//machina/tools/pip_package/utils:tf_wheel.bzl", "tf_wheel", "tf_wheel_dep")

package(default_visibility = ["//visibility:public"])

# This returns a list of headers of all public header libraries (e.g.,
# framework, lib), and all of the transitive dependencies of those
# public headers.  Not all of the headers returned by the filegroup
# are public (e.g., internal headers that are included by public
# headers), but the internal headers need to be packaged in the
# pip_package for the public headers to be properly included.
#
# Public headers are therefore defined by those that are both:
#
# 1) "publicly visible" as defined by bazel
# 2) Have documentation.
#
# This matches the policy of "public" for our python API.
transitive_hdrs(
    name = "included_headers",
    deps = [
        "//machina/compiler/tf2xla:xla_compiled_cpu_function",
        "//machina/compiler/mlir:mlir_graph_optimization_pass",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:stream_executor",
        "//machina/cc/saved_model:loader",
        "//machina/cc/saved_model:reader",
        "//machina/cc/saved_model:bundle_v2",
        "//machina/c/experimental/filesystem:filesystem_interface",
        "//machina/c/experimental/next_pluggable_device:c_api_hdrs",
        "//machina/c/experimental/pluggable_profiler:pluggable_profiler_hdrs",
        "//machina/c/experimental/stream_executor:stream_executor_hdrs",
        "//machina/c/experimental/grappler:grappler_hdrs",
        "//machina/c:kernels_hdrs",
        "//machina/c:ops_hdrs",
        "@local_xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "//machina/core/common_runtime/next_pluggable_device/c:plugin_c_api_hdrs",
        # TODO(rostam): Revisit these after cc_shared_library everywhere
        "//machina/cc:client_session",
        "//machina/cc:cc_ops",
        # WARNING: None of the C/C++ code under python/ has any API guarantees, and TF team
        # reserves the right to change APIs and other header-level interfaces.  If your custom
        # op uses these headers, it may break when users upgrade their version of machina.
        # NOTE(ebrevdo): None of the python symbols are exported by libmachina_framework.so.
        # Code that relies on these headers should dynamically link to
        # _pywrap_machina_internal.so as well.
        "//machina/python/grappler:model_analyzer_lib",
        "//machina/python/lib/core:py_exception_registry",
        "//machina/python/lib/core:pybind11_proto",
        "//machina/python/lib/core:pybind11_absl",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status",
        "//machina/python/lib/core:py_func_lib",
        "//machina/python/lib/core:py_seq_tensor",
        "//machina/python/lib/core:py_util",
        "//machina/python/util:cpp_python_util",
        "//machina/python/util:kernel_registry",
        "//machina/python/framework:python_op_gen",
        "//machina/python/client:tf_session_helper",
    ] + if_cuda([
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

filegroup(
    name = "xla_cmake",
    srcs = ["xla_build/CMakeLists.txt"],
)

filegroup(
    name = "licenses",
    srcs = [
        "//:LICENSE",
        "//third_party/icu/data:LICENSE",
        "@XNNPACK//:LICENSE",
        "@com_google_absl//:LICENSE",
        "@com_google_protobuf//:LICENSE",
        "@curl//:COPYING",
        "@ducc//:LICENSE",
        "@flatbuffers//:LICENSE",
        "@gemmlowp//:LICENSE",
        "@libjpeg_turbo//:LICENSE.md",
        "@toolchain-project//toolchain:LICENSE.TXT",
        "@toolchain-project//mlir:LICENSE.TXT",
        "@local_config_tensorrt//:LICENSE",
        "@ml_dtypes_py//:LICENSE",
        "@org_brotli//:LICENSE",
        "@pasta//:LICENSE",
        "@png//:LICENSE",
        "@ruy//:LICENSE",
        "@snappy//:COPYING",
        "@stablehlo//:LICENSE",
        "@tf_runtime//:LICENSE",
    ] + select({
        "//machina:android": [],
        "//machina:ios": [],
        "//machina:linux_s390x": [],
        "//machina:windows": [],
        "//conditions:default": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + if_cuda([
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "@local_xla//third_party/mkl_dnn:LICENSE",
    ]) + if_enable_mkl(["@local_xla//xla/tsl/mkl:LICENSE"]) + if_not_system_lib(
        "absl_py",
        [
            "@absl_py//:LICENSE",
        ],
    ) + tf_additional_license_deps(),
)

# Add dynamic kernel dso files here.
DYNAMIC_LOADED_KERNELS = [
    "//machina/core/kernels:libtfkernel_sobol_op.so",
]

COMMON_PIP_DEPS = [
    "//machina/compiler/mlir/stablehlo:stablehlo",
    "//machina/compiler/mlir/machina:gen_mlir_passthrough_op_py",
    "//machina/core/function/trace_type:serialization_test_proto_py",
    "//machina/core/function/trace_type:serialization",
    "//machina/dtensor/python/tests:multi_client_test_util",
    "//machina/dtensor/python/tests:test_util",
    "//machina/lite/python:tflite_convert",
    "//machina/lite/tools:visualize",
    "//machina/python/autograph/core:test_lib",
    "//machina/python/autograph/impl/testing:pybind_for_testing",
    "//machina/python/autograph/pyct/testing:basic_definitions",
    "//machina/python/autograph/pyct/testing:decorators",
    "//machina/python/autograph/pyct/common_transformers:common_transformers",
    "//machina/python/autograph/pyct/static_analysis:type_inference",
    "//machina/python/autograph/utils:context_managers",
    "//machina/python/autograph/utils:tensor_list",
    "//machina/python/compiler:compiler",
    "//machina/python/ops:cond_v2",
    "//machina/python:distributed_framework_test_lib",
    "//machina/python/distribute:distribute_test_lib_pip",
    "//machina/python/training/experimental:loss_scale",
    "//machina/python/training/experimental:loss_scale_optimizer",
    "//machina/python/data/benchmarks:benchmark_base",
    "//machina/python/data/experimental/kernel_tests/service:multi_process_cluster",
    "//machina/python/data/experimental/kernel_tests/service:test_base",
    "//machina/python/data/experimental/ops:testing",
    "//machina/python/data/experimental/service:server_lib",
    "//machina/python/data/kernel_tests:checkpoint_test_base",
    "//machina/python/data/kernel_tests:test_base",
    "//machina/python/data/kernel_tests:tf_record_test_base",
    "//machina/python/ops/ragged:ragged_tensor_test_ops",
    "//machina/python/debug:debug_pip",
    "//machina/python/distribute:combinations",
    "//machina/python/distribute/coordinator:fault_tolerance_test_base",
    "//machina/python/distribute/failure_handling:check_preemption_py",
    "//machina/python/distribute/failure_handling:failure_handling_lib",
    "//machina/python/distribute/failure_handling:failure_handling_util",
    "//machina/python/distribute:multi_process_runner",
    "//machina/python/eager:eager_pip",
    "//machina/python/keras:combinations",
    "//machina/python/keras/mixed_precision:test_util",
    "//machina/python/keras/utils:dataset_creator",
    "//machina/python/kernel_tests/nn_ops:bias_op_base",
    "//machina/python/kernel_tests/nn_ops:cudnn_deterministic_base",
    "//machina/python/kernel_tests/nn_ops:depthwise_conv_op_base",
    "//machina/python/kernel_tests/nn_ops:xent_op_test_base",
    "//machina/python/kernel_tests/random:util",
    "//machina/python/kernel_tests/signal:test_util",
    "//machina/python/kernel_tests/sparse_ops:sparse_xent_op_test_base",
    "//machina/python/lib:__init__",
    "//machina/python/ops/parallel_for:test_util",
    "//machina/python/ops/structured:structured_tensor_dynamic",
    "//machina/python/platform:resource_loader",
    "//machina/python/profiler:trace",
    "//machina/python/saved_model:saved_model",
    "//machina/python/summary:__init__",
    "//machina/python/summary:plugin_asset",
    "//machina/python/summary:summary_iterator",
    "//machina/python/summary:summary_py",
    "//machina/python/tools:tools_pip",
    "//machina/python/tools/api/generator:create_python_api",
    "//machina/python/tools/api/generator2/extractor:extractor",
    "//machina/python/tools/api/generator2/generator:generator",
    "//machina/python/tpu",
    "//machina/python/util:deprecated_module",
    "//machina/python/util:deprecated_module_new",
    "//machina/python/util:example_parser_configuration",
    "//machina/python/util:function_utils",
    "//machina/python/util:keyword_args",
    "//machina/python/util:lock_util",
    "//machina/python/util:module_wrapper",
    "//machina/python/util:serialization",
    "//machina/python/ops:image_grad_test_base",
    "//machina/python/framework:memory_checker",
    "//machina/python/framework:test_ops",
    "//machina/python/ops:while_v2",
    "//machina/tools/common:public_api",
    "//machina/tools/common:test_module1",
    "//machina/tools/common:traverse",
    "//machina/tools/docs:tf_doctest_lib",
    "@local_tsl//tsl/profiler/protobuf:trace_events_proto_py",
    "//machina/python/distribute:parameter_server_strategy_v2",
    "//machina/python/distribute/coordinator:cluster_coordinator",
    "//machina/python/distribute/coordinator:remote_eager_lib",
    "//machina/python/distribute/coordinator:metric_utils",
    "//machina/python/distribute/experimental/rpc:rpc_ops",
    "//machina/python/util:pywrap_xla_ops",
    "//machina:machina_py",
    "//machina/tools/compatibility:tf_upgrade_v2",
] + if_with_tpu_support([
    "//machina/python/tools:grpc_tpu_worker",
    "//machina/python/tools:grpc_tpu_worker_service",
])

py_binary(
    name = "build_pip_package_py",
    srcs = [
        "build_pip_package.py",
    ],
    main = "build_pip_package.py",
    deps = [
        "//machina/tools/pip_package/utils:py_utils",
        "@pypi_setuptools//:pkg",
        "@pypi_wheel//:pkg",
    ],
)

collect_data_files(
    name = "cc_deps",
    deps = [
        #`machina_py` is the main target to collect all required `data` dependencies
        "//machina:machina_py",
        # utils below are not part of machina_py DAG
        "//machina/compiler/mlir/stablehlo:stablehlo_extension",
        "//machina/python/autograph/impl/testing:pybind_for_testing",
        "//machina/python/framework:memory_checker",
        "//machina/python/util:pywrap_xla_ops",
    ],
)

transitive_py_deps(
    name = "py_deps",
    deps = COMMON_PIP_DEPS,
)

genrule(
    name = "setup_py",
    srcs = ["setup.py.tpl"],
    outs = ["setup.py"],
    cmd = """sed -E "s/_VERSION = '0.0.0'/_VERSION = '{wheel_version}{wheel_version_suffix}'/" \
$(location setup.py.tpl) > $@;""".format(
        wheel_version = TF_VERSION,
        wheel_version_suffix = TF_SEMANTIC_VERSION_SUFFIX,
    ),
)

py_binary(
    name = "setup_py_binary",
    srcs = [":setup_py"],
    data = [
        "//:AUTHORS",
        "//:LICENSE",
    ],
    main = "setup.py",
)

tf_wheel(
    name = "wheel",
    headers = [
        ":included_headers",
        ":licenses",
        "//machina/core:protos_all_proto_srcs",
    ],
    platform_name = select({
        "@platforms//os:osx": "macosx",
        "@platforms//os:macos": "macosx",
        "@platforms//os:windows": "win",
        "@platforms//os:linux": "linux",
    }),
    platform_tag = select({
        "@platforms//cpu:aarch64": "arm64",
        "@platforms//cpu:arm64": "arm64",
        "@platforms//cpu:x86_64": "x86_64",
        "@platforms//cpu:ppc": "ppc64le",
    }),
    source_files = [
        "MANIFEST.in",
        "//machina/tools/pip_package:THIRD_PARTY_NOTICES.txt",
        ":setup_py",
        ":cc_deps",
        ":py_deps",
        "//machina:tf_python_api_gen_v2",
    ] + select({
        "//machina:dynamic_loaded_kernels": if_pywrap(
            if_false = DYNAMIC_LOADED_KERNELS,
            if_true = [],
        ),
        "//conditions:default": [],
    }) + select({
        "//conditions:default": if_pywrap(
            if_false = [
                "//machina:machina_cc",
                "//machina:machina_framework",
            ],
            if_true = [
                "//machina/python:pywrap_machina_binaries",
            ],
        ),
        "//machina:windows": if_pywrap(
            if_false = [
                "//machina/python:pywrap_machina_import_lib_file",
            ],
            if_true = [
                "//machina/python:pywrap_machina_binaries",
            ],
        ),
    }) + if_mkl_ml(["@local_xla//xla/tsl/mkl:intel_binary_blob"]),
    xla_aot_compiled = [
        "//machina/compiler/tf2xla:xla_compiled_cpu_runtime_srcs",
        ":xla_cmake",
    ],
)

genrule(
    name = "empty_test",
    outs = ["empty_test.py"],
    cmd = "echo '' > $@",
)

py_test(
    name = "prebuilt_wheel_import_api_packages_test_cpu",
    srcs = if_wheel_dependency(
        ["import_api_packages_test.py"],
        [":empty_test"],
    ),
    main = if_wheel_dependency("import_api_packages_test.py", "empty_test.py"),
    tags = [
        "cpu",
        "manual",
    ],
    deps = if_wheel_dependency(tf_wheel_dep()) + [
        "@pypi_absl_py//:pkg",
    ],
)

py_test(
    name = "prebuilt_wheel_import_api_packages_test_gpu",
    srcs = if_wheel_dependency(
        ["import_api_packages_test.py"],
        [":empty_test"],
    ),
    exec_properties = if_cuda(
        tf_exec_properties({"tags": tf_cuda_tests_tags()}),
        {},
    ),
    main = if_wheel_dependency("import_api_packages_test.py", "empty_test.py"),
    tags = [
        "gpu",
        "manual",
    ],
    deps = if_wheel_dependency(tf_wheel_dep()) + [
        "@pypi_absl_py//:pkg",
    ],
)

py_test(
    name = "import_api_packages_test_cpu",
    srcs = ["import_api_packages_test.py"],
    main = "import_api_packages_test.py",
    tags = [
        "cpu",
        "manual",
    ],
    deps = [
        ":tf_py_import",
    ],
)

py_test(
    name = "import_api_packages_test_gpu",
    srcs = ["import_api_packages_test.py"],
    exec_properties = if_cuda(
        tf_exec_properties({"tags": tf_cuda_tests_tags()}),
        {},
    ),
    main = "import_api_packages_test.py",
    tags = [
        "gpu",
        "manual",
    ],
    deps = [
        ":tf_py_import",
    ],
)

verify_manylinux_compliance_test(
    name = "manylinux_compliance_test",
    aarch64_compliance_tag = "manylinux_2_17_aarch64",
    ppc64le_compliance_tag = "manylinux_2_17_ppc64le",
    test_tags = [
        "manual",
    ],
    wheel = ":wheel",
    x86_64_compliance_tag = "manylinux_2_27_x86_64",
)

py_import(
    name = "tf_py_import",
    wheel = ":wheel",
    wheel_deps = if_cuda([
        "@pypi_nvidia_cublas_cu12//:pkg",
        "@pypi_nvidia_cuda_cupti_cu12//:pkg",
        "@pypi_nvidia_cuda_nvrtc_cu12//:pkg",
        "@pypi_nvidia_cuda_runtime_cu12//:pkg",
        "@pypi_nvidia_cudnn_cu12//:pkg",
        "@pypi_nvidia_cufft_cu12//:pkg",
        "@pypi_nvidia_curand_cu12//:pkg",
        "@pypi_nvidia_cusolver_cu12//:pkg",
        "@pypi_nvidia_cusparse_cu12//:pkg",
        "@pypi_nvidia_nccl_cu12//:pkg",
        "@pypi_nvidia_nvjitlink_cu12//:pkg",
        "@pypi_nvidia_nvshmem_cu12//:pkg",
    ]),
    deps = [
        "@pypi_absl_py//:pkg",
        "@pypi_astunparse//:pkg",
        "@pypi_flatbuffers//:pkg",
        "@pypi_gast//:pkg",
        "@pypi_ml_dtypes//:pkg",
        "@pypi_numpy//:pkg",
        "@pypi_opt_einsum//:pkg",
        "@pypi_packaging//:pkg",
        "@pypi_protobuf//:pkg",
        "@pypi_requests//:pkg",
        "@pypi_termcolor//:pkg",
        "@pypi_typing_extensions//:pkg",
        "@pypi_wrapt//:pkg",
    ],
)
