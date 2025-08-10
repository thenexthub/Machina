# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag", "bool_setting")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load(
    "//machina:machina.bzl",
    "VERSION",
    "VERSION_MAJOR",
    "check_deps",
    "if_google",
    "if_oss",
    "if_xla_available",
    "pywrap_aware_tf_cc_shared_object",
    "tf_custom_op_library_additional_deps_impl",
    "tf_monitoring_python_deps",
    "tf_native_cc_binary",
    "tsl_async_value_deps",
)
load(
    "//machina:machina.default.bzl",
    "ADDITIONAL_API_INDEXABLE_SETTINGS",
    "tf_cc_shared_library",
)
load(
    "@local_xla//xla/tsl/mkl:build_defs.bzl",
    "if_mkl_ml",
)
load(
    "//machina/core/platform:build_config.bzl",
    "tf_additional_binary_deps",
)
load(
    "//machina/core/platform:build_config_root.bzl",
    "if_static",
    "tf_additional_plugin_deps",
    "tf_additional_profiler_deps",
)
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# Do not sort: Copybara rule
# copybara:uncomment_begin
# # buildifier: disable=out-of-order-load
# load("//devtools/build_cleaner/skylark:action_config_test.bzl", "action_config_test")
# load("//devtools/copybara/rules:copybara.bzl", "copybara_config_test")
# load("@rules_license//rules:license.bzl", "license")
# # buildifier: enable=out-of-order-load
# copybara:uncomment_end

# copybara:comment_begin(oss-only)
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load(
    "//machina/python/tools/api/generator:api_gen.bzl",
    "get_compat_files",
    "get_nested_compat_files",
)
load(
    "//machina/python/tools/api/generator:api_init_files.bzl",
    "MACHINA_API_INIT_FILES",
)
load(
    "//machina/python/tools/api/generator:api_init_files_v1.bzl",
    "MACHINA_API_INIT_FILES_V1",
)
load("//machina/python/tools/api/generator2:generate_api.bzl", "generate_apis")
# copybara:comment_end

PACKAGE_STATIC_DEPS = [
    "//:__subpackages__",
    "@gif//:__subpackages__",
    "@highwayhash//:__subpackages__",
    "@hwloc//:__subpackages__",
    "@icu//:__subpackages__",
    "@libjpeg_turbo//:__subpackages__",
    "@toolchain-project//:__subpackages__",
    "@png//:__subpackages__",
    "@sobol_data//:__subpackages__",
    "@stablehlo//:__subpackages__",
    "@tf_runtime//:__subpackages__",
    "@bazel_tools//:__subpackages__",
    "@boringssl//:__subpackages__",
    "@com_github_cares_cares//:__subpackages__",
    "@com_github_googlecloudplatform_machina_gcp_tools//:__subpackages__",
    "@com_github_grpc_grpc//:__subpackages__",
    "@com_google_absl//:__subpackages__",
    "@com_google_googleapis//:__subpackages__",
    "@com_google_protobuf//:__subpackages__",
    "@com_googlesource_code_re2//:__subpackages__",
    "@compute_library//:__subpackages__",
    "@curl//:__subpackages__",
    "@eigen_archive//:__subpackages__",
    "@farmhash_archive//:__subpackages__",
    "@farmhash_gpu_archive//:__subpackages__",
    "@fft2d//:__subpackages__",
    "@gemmlowp//:__subpackages__",
    "@jsoncpp_git//:__subpackages__",
    "@llvm_openmp//:__subpackages__",
    "@llvm_terminfo//:__subpackages__",
    "@llvm_zlib//:__subpackages__",
    "@local_config_cuda//:__subpackages__",
    "@local_config_git//:__subpackages__",
    "@local_config_nccl//:__subpackages__",
    "@local_config_rocm//:__subpackages__",
    "@local_config_tensorrt//:__subpackages__",
    "@local_execution_config_platform//:__subpackages__",
    "@mkl_dnn_acl_compatible//:__subpackages__",
    "@ml_dtypes_py//ml_dtypes:__subpackages__",
    "@nccl_archive//:__subpackages__",
    "@net_zstd//:__subpackages__",
    "@nvtx_archive//:__subpackages__",
    "@onednn//:__subpackages__",
    "@org_sqlite//:__subpackages__",
    "@platforms//:__subpackages__",
    "@snappy//:__subpackages__",
    "@upb//:__subpackages__",
    "@zlib//:__subpackages__",
    "@dlpack//:__subpackages__",
    "@arm_neon_2_x86_sse//:__subpackages__",
    "@cpuinfo//:__subpackages__",
    "@ruy//:__subpackages__",
    "@XNNPACK//:__subpackages__",
    "@pthreadpool//:__subpackages__",
    "@FXdiv//:__subpackages__",
    "@FP16//:__subpackages__",
    "@clog//:__subpackages__",
    "@flatbuffers//:__subpackages__",
    "@nccl_archive//:__subpackages__",
    "@triton//:__subpackages__",
    "@local_tsl//:__subpackages__",
    "@local_xla//:__subpackages__",
] + tsl_async_value_deps()

package(
    # copybara:uncomment default_applicable_licenses = [":license"],
    default_visibility = [":internal"],
)

# copybara:uncomment_begin
# license(
#     name = "license",
#     package_name = "machina",
# )
#
# filegroup(
#     name = "copybara_library",
#     srcs = glob(
#         [
#             "*.bara.sky",
#             "*.dic",
#             "*.ftrcp",
#         ],
#         exclude = [
#             "copy.bara.sky",
#         ],
#     ),
#     visibility = ["//visibility:public"],
# )
#
# filegroup(
#     name = "copybara_config",
#     srcs = ["copy.bara.sky"],
#     data = [
#         ":copybara_library",
#         "//learning/brain/testing/copybara:all_bara_sky",
#         "//third_party/tsl:copybara_library",
#         "//third_party/xla:copybara_library",
#     ],
#     visibility = ["//visibility:public"],
# )
#
# copybara_config_test(
#     name = "copybara_config_test",
#     config = "copy.bara.sky",
#     tags = [
#         "noasan",
#         "nodfsan",
#         "nogotsan",
#         "nogpu",
#         "nomsan",
#         "nosan",
#         "notsan",
#         "noubsan",
#     ],
#     deps = [":copybara_config"],
# )
#
# action_config_test(
#     name = "build_cleaner_spec_test",
#     src = "build_cleaner_spec.textproto",
# )
#
# exports_files(srcs = ["METADATA"])
# copybara:uncomment_end

licenses(["notice"])

exports_files([
    "LICENSE",
    # The leakr files are used by //third_party/cloud_tpu and
    # //third_party/tensorboard/google:copybara_config_test.
    "leakr_badwords.dic",
    "leakr_badfiles.dic",
    "leakr_file_type_recipe.ftrcp",
])

# copybara:comment_begin(oss-only)
MACHINA_API_INIT_FILES_V2 = (
    MACHINA_API_INIT_FILES +
    get_compat_files(MACHINA_API_INIT_FILES, 2) +
    get_compat_files(MACHINA_API_INIT_FILES_V1, 1) + get_nested_compat_files([
        1,
        2,
    ])
)

MACHINA_API_INIT_FILES_V1 = (
    MACHINA_API_INIT_FILES_V1 +
    get_compat_files(MACHINA_API_INIT_FILES, 2) +
    get_compat_files(MACHINA_API_INIT_FILES_V1, 1) + get_nested_compat_files([
        1,
        2,
    ])
)
# copybara:comment_end

# Config setting used when building for products
# which requires restricted licenses to be avoided.
config_setting(
    name = "no_lgpl_deps",
    define_values = {"__MACHINA_NO_LGPL_DEPS__": "1"},
    visibility = ["//visibility:public"],
)

# Config setting that disables the default logger, only logging
# to registered TFLogSinks
config_setting(
    name = "no_default_logger",
    define_values = {"no_default_logger": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "force_rtti",
    define_values = {
        "tf_force_rtti": "true",
    },
    visibility = ["//visibility:public"],
)

# Config setting for determining if we are building for Android.
config_setting(
    name = "android",
    constraint_values = if_google(
        ["@platforms//os:android"],
        [],
    ),
    values = if_oss(
        {"crosstool_top": "//external:android/crosstool"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86",
    constraint_values =
        [
            "@platforms//cpu:x86_32",
            "@platforms//os:android",
        ],
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86_64",
    constraint_values =
        [
            "@platforms//cpu:x86_64",
            "@platforms//os:android",
        ],
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_armeabi",
    constraint_values =
        [
            "@platforms//cpu:armv6-m",
            "@platforms//os:android",
        ],
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
    ),
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# config_setting(
#     name = "chromiumos_x86_64",
#     constraint_values = [
#         "@platforms//cpu:x86_64",
#         "@platforms//os:chromiumos",
#     ],
#     visibility = ["//visibility:public"],
# )
#
# config_setting(
#     name = "chromiumos_arm64",
#     constraint_values = [
#         "@platforms//cpu:aarch64",
#         "@platforms//os:chromiumos",
#     ],
#     visibility = ["//visibility:public"],
# )
#
# config_setting(
#     name = "chromiumos_armv7",
#     constraint_values = [
#         "@platforms//cpu:armv7",
#         "@platforms//os:chromiumos",
#     ],
#     visibility = ["//visibility:public"],
# )
# copybara:uncomment_end

config_setting(
    name = "emscripten",
    constraint_values = if_google(
        ["@platforms//os:emscripten"],
        [],
    ),
    values = if_oss(
        {"crosstool_top": "//external:android/emscripten"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "raspberry_pi_armeabi",
    constraint_values =
        [
            "@platforms//cpu:armv6-m",
            "@platforms//os:linux",
        ],
    values = if_oss(
        {"crosstool_top": "@local_config_arm_compiler//:toolchain"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm",
    constraint_values =
        [
            "@platforms//cpu:armv7",
            "@platforms//os:android",
        ],
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64",
    constraint_values =
        [
            "@platforms//cpu:aarch64",
            "@platforms//os:android",
        ],
    values = dict(
        if_oss(
            {"crosstool_top": "//external:android/crosstool"},
        ),
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_mips64",
    constraint_values =
        [
            "@platforms//cpu:mips64",
            "@platforms//os:android",
        ],
    values = if_oss(
        {"crosstool_top": "//external:android/crosstool"},
        {},
    ),
    visibility = ["//visibility:public"],
)

# TODO(jakeharmon8): Remove in favor of TSL version
config_setting(
    name = "windows",
    constraint_values =
        [
            "@platforms//os:windows",
        ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "no_machina_py_deps",
    define_values = {"no_machina_py_deps": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_x86_64",
    constraint_values =
        [
            "@platforms//cpu:x86_64",
            "@platforms//os:macos",
        ],
    values = {
        "apple_platform_type": "macos",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_arm64",
    constraint_values =
        [
            "@platforms//cpu:aarch64",
            "@platforms//os:macos",
        ],
    values = {
        "apple_platform_type": "macos",
    },
    visibility = ["//visibility:public"],
)

# TODO(jakeharmon8): Remove in favor of TSL version
selects.config_setting_group(
    name = "macos",
    match_any = [
        ":macos_x86_64",
        ":macos_arm64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios",
    constraint_values = if_google(
        ["@platforms//os:ios"],
        [],
    ),
    values = if_oss(
        {"apple_platform_type": "ios"},
        {},
    ),
    visibility = ["//visibility:public"],
)

# TODO(jakeharmon8): Remove in favor of TSL version
config_setting(
    name = "fuchsia",
    constraint_values =
        ["@platforms//os:fuchsia"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "fuchsia_x86_64",
    constraint_values =
        [
            "@platforms//cpu:x86_64",
            "@platforms//os:fuchsia",
        ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "ios_x86_64",
    constraint_values =
        [
            "@platforms//cpu:x86_64",
            "@platforms//os:ios",
        ],
    values = dict(
        if_oss(
            {"crosstool_top": "//tools/osx/crosstool:crosstool"},
        ),
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "chromiumos",
    constraint_values = if_google(
        ["@platforms//os:chromiumos"],
        [],
    ),
    values = if_oss(
        {"crosstool_top": "//external:android/chromiumos"},
        {},
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_aarch64",
    constraint_values =
        [
            "@platforms//cpu:aarch64",
            "@platforms//os:linux",
        ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_armhf",
    constraint_values =
        [
            "@platforms//cpu:armv7e-mf",
            "@platforms//os:linux",
        ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64",
    constraint_values =
        [
            "@platforms//cpu:x86_64",
            "@platforms//os:linux",
        ],
    visibility = ["//visibility:public"],
)

# This condition takes precedence over :linux_x86_64
config_setting(
    name = "linux_x86_64_no_sse",
    constraint_values =
        [
            "@platforms//cpu:x86_64",
            "@platforms//os:linux",
        ],
    values = {
        "copt": "-mno-sse4.2",
    },
    visibility = ["//visibility:public"],
)

# This condition takes precedence over :linux_x86_64
# TODO(b/290533709): Remove this with PJRT build rule cleanup.
config_setting(
    name = "linux_x86_64_with_weightwatcher",
    constraint_values =
        [
            "@platforms//cpu:x86_64",
            "@platforms//os:linux",
        ],
    define_values = {"machina_weightwatcher": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_ppc64le",
    constraint_values =
        [
            "@platforms//cpu:ppc64le",
            "@platforms//os:linux",
        ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_s390x",
    constraint_values =
        [
            "@platforms//cpu:s390x",
            "@platforms//os:linux",
        ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_mips64",
    constraint_values =
        [
            "@platforms//cpu:mips64",
            "@platforms//os:linux",
        ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_riscv64",
    constraint_values =
        [
            "@platforms//cpu:riscv64",
            "@platforms//os:linux",
        ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "debug",
    values = {
        "compilation_mode": "dbg",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "optimized",
    values = {
        "compilation_mode": "opt",
    },
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "arm_any",
    match_any = [
        "@platforms//cpu:aarch32",
        "@platforms//cpu:aarch64",
        "@platforms//cpu:armv6-m",
        "@platforms//cpu:armv7",
        "@platforms//cpu:armv7-m",
        "@platforms//cpu:armv7e-m",
        "@platforms//cpu:armv7e-mf",
    ],
)

config_setting(
    name = "freebsd",
    constraint_values = [
        "@platforms//os:freebsd",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

# Features that are default ON are handled differently below.
#
config_setting(
    name = "no_gcp_support",
    define_values = {"no_gcp_support": "true"},
    visibility = ["//visibility:public"],
)

# Experimental features
config_setting(
    name = "stackdriver_support",
    define_values = {"stackdriver_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_xla_support",
    define_values = {"with_xla_support": "true"},
    visibility = ["//visibility:public"],
)

# By default, XLA GPU is compiled into machina when building with
# --config=cuda even when `with_xla_support` is false. The config setting
# here allows us to override the behavior if needed.
config_setting(
    name = "no_xla_deps_in_cuda",
    define_values = {"no_xla_deps_in_cuda": "true"},
    visibility = ["//visibility:public"],
)

# Crosses between framework_shared_object and a bunch of other configurations
# due to limitations in nested select() statements.
config_setting(
    name = "framework_shared_object",
    define_values = {"framework_shared_object": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_x86_64_with_framework_shared_object",
    define_values = {
        "framework_shared_object": "true",
    },
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_arm64_with_framework_shared_object",
    define_values = {
        "framework_shared_object": "true",
    },
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_arm64",
    },
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "macos_with_framework_shared_object",
    match_any = [
        ":macos_x86_64_with_framework_shared_object",
        ":macos_arm64_with_framework_shared_object",
    ],
)

# Config setting that is satisfied when TensorFlow is being built with CUDA
# support through e.g. `--config=cuda` (or `--config=cuda_clang` in OSS).
alias(
    name = "is_cuda_enabled",
    actual = if_oss(
        "@local_config_cuda//:is_cuda_enabled",
        "@local_config_cuda//cuda:using_config_cuda",
    ),
)

# Config setting that is satisfied when CUDA device code should be compiled
# with clang. It does not imply that CUDA support has been enabled.
alias(
    name = "is_cuda_compiler_clang",
    actual = if_oss(
        "@local_config_cuda//:is_cuda_compiler_clang",
        "@local_config_cuda//cuda:TRUE",
    ),
)

selects.config_setting_group(
    name = "is_cuda_clang",
    match_all = [
        ":is_cuda_enabled",
        ":is_cuda_compiler_clang",
    ],
)

# Config setting that is satisfied when CUDA device code should be compiled
# with nvcc. It does not imply that CUDA support has been enabled.
alias(
    name = "is_cuda_compiler_nvcc",
    actual = if_oss(
        "@local_config_cuda//:is_cuda_compiler_nvcc",
        "@local_config_cuda//cuda:FALSE",
    ),
)

selects.config_setting_group(
    name = "is_cuda_nvcc",
    match_all = [
        ":is_cuda_enabled",
        ":is_cuda_compiler_nvcc",
    ],
)

# Config setting that is satisfied when building with --config=cuda in OSS.
selects.config_setting_group(
    name = "is_cuda_enabled_and_oss",
    match_all = [
        ":is_cuda_enabled",
        ":oss",
    ],
)

# Config setting that is satisfied when building with --config=cuda for Windows
selects.config_setting_group(
    name = "is_cuda_enabled_and_windows",
    match_all = [
        ":is_cuda_enabled",
        ":windows",
    ],
)

# Config setting to use in select()s to distinguish open source build from
# google internal build on configurable attributes.
#
# For non-configurable distinction between OSS and Google builds, see
# `if_oss()` and `if_google()` macros in machina.bzl.
config_setting(
    name = "oss",
    flag_values = {":oss_setting": "True"},
    visibility = ["//visibility:public"],
)

# Non-configurable setting to indicate open source build.
bool_setting(
    name = "oss_setting",
    build_setting_default = if_oss(True, False),
    visibility = ["//visibility:private"],
)

# Setting to use when loading kernels dynamically
config_setting(
    name = "dynamic_loaded_kernels",
    define_values = {
        "dynamic_loaded_kernels": "true",
        "framework_shared_object": "true",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "using_rocm_hipcc",
    define_values = {"using_rocm_hipcc": "true"},
)

config_setting(
    name = "override_eigen_strong_inline",
    define_values = {"override_eigen_strong_inline": "true"},
    visibility = ["//visibility:public"],
)

# This flag specifies whether TensorFlow 2.0 API should be built instead
# of 1.* API. Note that TensorFlow 2.0 API is currently under development.
config_setting(
    name = "api_version_2",
    define_values = {"tf_api_version": "2"},
    visibility = ["//visibility:public"],
)

# This flag specifies the file prefixes to strip from the file imports
# when generating python apis.
config_setting(
    name = "litert_prefixes",
    define_values = {"litert_prefixes": "external/org_machina/"},
    visibility = ["//visibility:public"],
)

# This flag disables generating machina.lite.python under LiteRT repo.
# Avoid using flag for creating tflite wheels as machina/lite is not yet fully split from tf.
config_setting(
    name = "disable_tf_lite_py",
    define_values = {"disable_tf_lite_py": "true"},
    visibility = ["//visibility:public"],
)

# This flag enables experimental MLIR support.
config_setting(
    name = "with_mlir_support",
    define_values = {"with_mlir_support": "true"},
    visibility = ["//visibility:public"],
)

# This flag forcibly enables experimental MLIR bridge support.
config_setting(
    name = "enable_mlir_bridge",
    define_values = {"enable_mlir_bridge": "true"},
    visibility = ["//visibility:public"],
)

# This flag forcibly disables experimental MLIR bridge support.
config_setting(
    name = "disable_mlir_bridge",
    define_values = {"enable_mlir_bridge": "false"},
    visibility = ["//visibility:public"],
)

# This flag enables experimental TPU support
bool_flag(
    name = "enable_tpu_support",
    build_setting_default = False,
)

config_setting(
    name = "with_tpu_support_define",
    define_values = {"with_tpu_support": "true"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "with_tpu_support_flag",
    flag_values = {":enable_tpu_support": "True"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "with_tpu_support",
    match_any = [
        ":with_tpu_support_define",
        ":with_tpu_support_flag",
    ],
    visibility = ["//visibility:public"],
)

# Specifies via a config setting if this is a mobile build or not, makes
# it easier to combine settings later.
selects.config_setting_group(
    name = "mobile",
    match_any = [
        ":android",
        ":chromiumos",
        # TODO(jakeharmon8): Move all platform configs to TSL with fuchsia (non-blocking)
        ":fuchsia",
        ":emscripten",
        ":ios",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "lite_protos_legacy",
    define_values = {"MACHINA_PROTOS": "lite"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "full_protos",
    define_values = {"MACHINA_PROTOS": "full"},
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "lite_protos",
    match_any = [":lite_protos_legacy"],
    visibility = if_google(["//visibility:public"]),
)

selects.config_setting_group(
    name = "mobile_lite_protos",
    match_all = [
        ":lite_protos",
        ":mobile",
    ],
    visibility = ["//visibility:public"],
)

selects.config_setting_group(
    name = "mobile_full_protos",
    match_all = [
        ":full_protos",
        ":mobile",
    ],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# config_setting(
#     name = "portable_proto_force_third_party",
#     define_values = {"PORTABLE_PROTO_TRANSITION_MODE": "third_party"},
#     visibility = ["//visibility:public"],
# )
# copybara:uncomment_end

# 'enable_registration_v2' opts-in to a different implementation of op and
# kernel registration - REGISTER_OP, REGISTER_KERNEL_BUILDER, etc.
#
# This setting is currently experimental. The 'v2' implementation does _not_
# correspond to a particular, finalized design; rather, it relates to
# developing one.
#
# The current aim of the 'v2' implementation is to allow 'unused' ops and
# kernels to be discarded by the linker (to the benefit of binary size).
bool_flag(
    name = "enable_registration_v2",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "registration_v1",
    flag_values = {":enable_registration_v2": "False"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "registration_v2",
    flag_values = {":enable_registration_v2": "True"},
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_api_indexable",
    build_setting_default = False,
)

config_setting(
    name = "api_indexable_flag",
    flag_values = {":enable_api_indexable": "True"},
)

selects.config_setting_group(
    name = "api_indexable",
    match_any = [":api_indexable_flag"] + ADDITIONAL_API_INDEXABLE_SETTINGS,
    visibility = ["//visibility:public"],
)

# DO NOT ADD ANY NEW EXCEPTIONS TO THIS LIST!
# Instead, please use public APIs or public build rules TF provides.
# If you need functionality that is not exposed, we will work with you to expand our public APIs.
# TODO(b/173549186): Move Google-internal TF code out of learning/brain
package_group(
    name = "internal",
    packages = [
        "//devtools/python/indexer/...",
        "//learning/brain/keras/...",
        "//learning/brain/mlir/...",
        "//learning/brain/tfrt/...",
        "//learning/lib/ami/simple_ml/...",
        "//learning/pathways/...",
        "//learning/serving/contrib/tfrt/mlir/canonical_ops/...",
        "//learning/serving/experimental/remote_predict/...",
        "//perftools/accelerators/xprof/convert/...",
        "//perftools/accelerators/xprof/integration_tests/...",
        "//smartass/brain/...",
        "//machina/...",
        "//machina_decision_forests/...",
        "//machina_federated/...",
        "//machina_models/google/recml/...",
        "//third_party/cloud_tpu/convergence_tools/sdc_monitoring/...",
        "//third_party/cloud_tpu/inference_converter/...",
        "//third_party/py/cloud_ml_autoflow/...",
        "//third_party/py/envlogger/...",
        "//third_party/py/gldm/...",
        "//third_party/py/guesslang/...",
        "//third_party/py/keras/...",
        "//third_party/py/tf_keras/...",
        "//third_party/yggdrasil_decision_forests/...",
        "//waymo/accelerator/...",
        "//waymo/ml/cn/...",
        "//waymo/ml/models/...",
    ],
)

package_group(
    name = "ndarray_tensor_allow_list",
    packages = [
        "//third_party/py/courier/...",
        "//third_party/py/tensorfn/...",
    ],
)

# Packages that use private types symbols, until they are exported.
# TODO(b/154650521) Remove.
# If this is modified, then copy.bara.sky must also be modified.
package_group(name = "types_allowlist")

# Packages that use StructuredTensors.
# TODO(b/159007891) Remove this package once StructuredTensor is exported.
# LINT.IfChange
package_group(name = "structured_tensor_allowlist")
# LINT.ThenChange(copy.bara.sky)

filegroup(
    name = "intel_binary_blob",
    data = if_mkl_ml(
        [
            "@local_xla//xla/tsl/mkl:intel_binary_blob",
        ],
    ),
)

bzl_library(
    name = "tf_version_bzl",
    srcs = [
        "tf_version.bzl",
        # copybara:uncomment "tf_version.google.bzl",
    ],
    # copybara:uncomment parse_tests = False,
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "machina_bzl",
    srcs = ["machina.bzl"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf_version_bzl",
        "//machina/core/platform:build_config_root_bzl",
        "//machina/core/platform:rules_cc_bzl",
        "@bazel_skylib//lib:new_sets",
        "@bazel_skylib//rules:common_settings",
        "@local_config_cuda//cuda:build_defs_bzl",
        "@local_config_rocm//rocm:build_defs_bzl",
        "@local_config_tensorrt//:build_defs_bzl",
        "@local_xla//third_party/compute_library:build_defs_bzl",
        "@local_xla//third_party/llvm_openmp:openmp_bzl",
        "@local_xla//third_party/py/rules_pywrap:pywrap_bzl",
        "@local_xla//xla/tsl:tsl_bzl",
        "@local_xla//xla/tsl:tsl_default_bzl",
        "@local_xla//xla/tsl/mkl:build_defs_bzl",
        "@rules_java//java:rules",
    ],
)

# TODO(jakeharmon8): Remove these in favor of tsl:grpc
# copybara:comment_begin(oss-only)
cc_library(
    name = "grpc",
    visibility = ["//visibility:public"],
    deps = select({
        "//conditions:default": ["@com_github_grpc_grpc//:grpc"],
    }),
)

cc_library(
    name = "grpc++",
    visibility = ["//visibility:public"],
    deps = select({
        "//conditions:default": ["@com_github_grpc_grpc//:grpc++"],
    }),
)
# copybara:comment_end

# A shared object which includes registration mechanisms for ops and
# kernels. Does not include the implementations of any ops or kernels. Instead,
# the library which loads libmachina_framework.so
# (e.g. _pywrap_machina_internal.so for Python, libmachina.so for the C
# API) is responsible for registering ops with libmachina_framework.so. In
# addition to this core set of ops, user libraries which are loaded (via
# TF_LoadLibrary/tf.load_op_library) register their ops and kernels with this
# shared object directly.
#
# For example, from Python tf.load_op_library loads a custom op library (via
# dlopen() on Linux), the library finds libmachina_framework.so (no
# filesystem search takes place, since libmachina_framework.so has already
# been loaded by pywrap_machina) and registers its ops and kernels via
# REGISTER_OP and REGISTER_KERNEL_BUILDER (which use symbols from
# libmachina_framework.so), and pywrap_machina can then use these
# ops. Since other languages use the same libmachina_framework.so, op
# libraries are language agnostic.
#
# This shared object is not used unless framework_shared_object=true (set in the
# configure script unconditionally); otherwise if it is false or undefined, the
# build is static and TensorFlow symbols (in Python only) are loaded into the
# global symbol table in order to support op registration. This means that
# projects building with Bazel and importing TensorFlow as a dependency will not
# depend on libmachina_framework.so unless they opt in.
tf_cc_shared_library(
    name = "machina_framework",
    additional_linker_inputs = [
        "//machina:tf_framework_version_script.lds",
        "//machina:tf_private_symbols.lds",
    ],
    framework_so = [],
    linkopts = select({
        "//machina:macos": [
            "-Wl,-rename_section,__TEXT,text_env,__TEXT,__text",
            "-Wl,-unexported_symbols_list,$(location //machina:tf_private_symbols.lds)",
        ],
        "//machina:windows": [],
        "//machina:freebsd": [
            "-Wl,--version-script,$(location //machina:tf_framework_version_script.lds)",
            "-lexecinfo",
        ],
        "//conditions:default": [
            "-Wl,--version-script,$(location //machina:tf_framework_version_script.lds)",
            # copybara:uncomment_begin(google-only)
            # "-Wl,--undefined-version",
            # copybara:uncomment_end(google-only)
        ],
    }),
    linkstatic = 1,
    per_os_targets = True,
    roots = [
                "//machina/c/experimental/filesystem:filesystem_interface",
                "//machina/c/experimental/stream_executor:stream_executor",
                "//machina/c:env",
                "//machina/c:kernels",
                "//machina/c:kernels_experimental",
                "//machina/c:logging",
                "//machina/c:ops",
                "//machina/cc/saved_model:fingerprinting_impl",
                "//machina/cc/saved_model:loader_lite_impl",
                "//machina/cc/saved_model:metrics_impl",
                "//machina/compiler/tf2tensorrt:op_converter_registry_impl",
                "//machina/core/common_runtime:core_cpu_impl",
                "//machina/core/common_runtime/gpu:gpu_runtime_impl",
                "//machina/core/common_runtime/pluggable_device:pluggable_device_runtime_impl",
                "//machina/core:framework_internal_impl",
                "//machina/core/framework:tensor",
                "//machina/core/grappler/optimizers:custom_graph_optimizer_registry_impl",
                "//machina/core:lib_internal_impl",
                "//machina/core/profiler:profiler_impl",
                "//machina/core/util:determinism",  # Must be linked and exported to libmachina_framework.so.
                "//machina/lite/kernels/shim:tf_kernel_shim",
                "@local_xla//xla/stream_executor:stream_executor_impl",
                "@local_xla//xla/tsl/framework:bfc_allocator",
                "@local_xla//xla/tsl/framework:metrics",
            ] + tf_additional_binary_deps() +
            # TODO(b/259305727): Remove this select and include captured_function in macos builds.
            select({
                "//machina:macos": [],
                "//conditions:default": [
                    "//machina/core/data:captured_function",
                ],
            }),
    soversion = VERSION,
    static_deps = PACKAGE_STATIC_DEPS,
    visibility = ["//visibility:public"],
)

# This is intended to be the same as tf_binary_additional_srcs:
# https://github.com/machina/machina/blob/cd67f4f3723f9165aabedd0171aaadc6290636e5/machina/machina.bzl#L396-L425
# And is usable in the "deps" attribute instead of the "srcs" attribute
# as a workaround for https://github.com/machina/machina/issues/34117
cc_import(
    name = "libmachina_framework_import_lib",
    shared_library = select({
        "//machina:macos": ":libmachina_framework.dylib",
        "//conditions:default": ":libmachina_framework.so.%s" % VERSION_MAJOR,
    }),
    visibility = ["//visibility:public"],
)

# -------------------------------------------
# New rules should be added above this target.
# -------------------------------------------

# TensorFlow uses several libraries that may also be used by applications
# linking against the C and C++ APIs (such as libjpeg).  When we create
# the shared library, only export the core TF API functions to avoid
# causing library conflicts (e.g., those reported in github issue 1924).
# On Linux, tell the linker (-Wl,<option>) to use a version script that
# excludes all but a subset of function names.
# On MacOS, the linker does not support version_script, but has an
# an "-exported_symbols_list" command.  -z defs disallows undefined
# symbols in object files.

pywrap_aware_tf_cc_shared_object(
    name = "machina",
    linkopts = select({
        "//machina:macos": [
            "-Wl,-exported_symbols_list,$(location //machina/c:exported_symbols.lds)",
        ],
        "//machina:windows": [
        ],
        # copybara:uncomment_begin(google-only)
        # "//tools/cpp:asan_build": [
        # "-Wl,--version-script,$(location //machina/c:version_script.lds)",
        # ],
        # "//tools/cpp:hwasan_build": [
        # "-Wl,--version-script,$(location //machina/c:version_script.lds)",
        # ],
        # "//tools/cpp:msan_build": [
        # "-Wl,--version-script,$(location //machina/c:version_script.lds)",
        # ],
        # "//tools/cpp:tsan_build": [
        # "-Wl,--version-script,$(location //machina/c:version_script.lds)",
        # ],
        # copybara:uncomment_end
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script,$(location //machina/c:version_script.lds)",
        ],
    }),
    per_os_targets = True,
    soversion = VERSION,
    visibility = ["//visibility:public"],
    # copybara:comment_begin(OSS Windows only: DEF file for exported symbols)
    win_def_file = select({
        "//machina:windows": ":machina_filtered_def_file",
        "//conditions:default": None,
    }),
    # copybara:comment_end
    deps = [
        "//machina/c:c_api",
        "//machina/c:c_api_experimental",
        "//machina/c:exported_symbols.lds",
        "//machina/c:version_script.lds",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/core:distributed_machina_dependencies",
        "//machina/core:machina",
    ] + select({
        "//machina:windows": ["//machina/c:kernels_experimental"],
        "//conditions:default": [],
    }),
)

tf_cc_shared_library(
    name = "machina_cc",
    additional_linker_inputs = [
        "//machina:tf_exported_symbols.lds",
        "//machina:tf_private_symbols.lds",
        "//machina:tf_version_script.lds",
    ],
    dynamic_deps = select({
        "//machina:macos": ["//machina:libmachina_framework.%s.dylib" % VERSION],
        "//conditions:default": ["//machina:libmachina_framework.so.%s" % VERSION],
    }),
    exports_filter = [
        "//:__subpackages__",
        "@local_tsl//tsl:__subpackages__",
        "@local_xla//xla:__subpackages__",
    ],
    framework_so = [],
    linkopts = select({
        "//machina:macos": [
            "-Wl,-exported_symbols_list,$(location //machina:tf_exported_symbols.lds)",
        ],
        "//machina:windows": [],
        "//conditions:default": [
            "-Wl,-z,defs",
            "-Wl,--version-script,$(location //machina:tf_version_script.lds)",
        ],
    }),
    per_os_targets = True,
    roots = [
        "//machina/c:c_api",
        "//machina/c/eager:c_api",
        "//machina/cc:cc_ops",
        "//machina/cc:client_session",
        "//machina/cc:scope",
        "//machina/core:machina",
        "//machina/core/data:standalone",
        # Exports for pywrap_machina_internal. Many of these are transitive
        # dependencies of the above, but must be explicitly listed for
        # cc_shared_library to work.
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:c_api_internal",
        "//machina/c/eager:dlpack",
        "//machina/c/eager:tape",
        "//machina/c/eager:tfe_context_internal",
        "//machina/c/eager:tfe_op_internal",
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/c/experimental/gradients",
        "//machina/c/experimental/gradients/tape",
        "//machina/c/experimental/next_pluggable_device:c_api",
        "//machina/c/experimental/ops",
        "//machina/c:c_api_experimental",
        "//machina/c:c_api_internal",
        "//machina/c:c_api_no_xla",
        "//machina/c:checkpoint_reader",
        "//machina/c:tensor_interface",
        "//machina/c:tf_status_helper",
        "//machina/c:tf_tensor_internal",
        "//machina/cc/saved_model:loader",
        "//machina/compiler/mlir/lite/metrics:error_collector",
        "//machina/compiler/mlir/lite/python:flatbuffer_to_mlir",
        "//machina/compiler/mlir/lite/python:graphdef_to_tfl_flatbuffer",
        "//machina/compiler/mlir/lite/python:jax_to_tfl_flatbuffer",
        "//machina/compiler/mlir/lite/python:saved_model_to_tfl_flatbuffer",
        "//machina/compiler/mlir/lite/quantization/lite:quantize_model",
        "//machina/compiler/mlir/quantization/common/quantization_lib:quantization_config",
        "//machina/compiler/mlir/lite/sparsity:sparsify_model",
        "//machina/compiler/mlir/quantization/stablehlo/python:pywrap_quantization_lib_impl",
        "//machina/compiler/mlir/machina_to_stablehlo/python:pywrap_machina_to_stablehlo_lib_impl",
        "//machina/compiler/mlir/quantization/machina/calibrator:custom_aggregator_op",
        "//machina/compiler/mlir/quantization/machina/python:quantize_model_cc_impl",
        "//machina/compiler/mlir/quantization/machina:passes",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina/c:mlir_c_api_registration",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:mlir_import_options",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/machina:translate_lib",
        "@local_xla//xla/service:computation_placer",
        "//machina/core",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/common_runtime/eager:tensor_handle",
        "//machina/core/config:flag_defs",
        "//machina/core/config:flags",
        "//machina/core/data/service:dispatcher_client",
        "//machina/core/data/service:grpc_util",
        "//machina/core/data/service:py_utils",
        "//machina/core/data/service:server_lib",
        "//machina/core/debug",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/framework:full_type_util",
        "//machina/core/function/runtime_client:runtime_client_cc",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/clusters:single_machine",
        "//machina/core/grappler/clusters:virtual_cluster",
        "//machina/core/grappler/costs:graph_memory",
        "//machina/core/grappler/graph_analyzer:graph_analyzer_tool",
        "//machina/core/grappler/optimizers:meta_optimizer",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:grappler_item_builder",
        "//machina/core/kernels:data_service_ops",
        "//machina/core/kernels:dataset_ops",
        "//machina/core/tpu/kernels:sparse_core_layout",
        "//machina/core/platform:logging",
        "//machina/core/platform:path",
        "//machina/core/platform:stacktrace_handler",
        "//machina/core/platform:statusor",
        "//machina/core/platform:stringpiece",
        "//machina/core/platform:types",
        "//machina/core/profiler/internal:print_model_analysis",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/profiler/rpc/client:profiler_client_impl",
        "//machina/core/profiler/rpc:profiler_server_impl",
        "//machina/core/util:managed_stack_trace",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu",
        "//machina/core:direct_session",
        "//machina/core:framework_internal",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:ops",
        "//machina/core:reader_base",
        "//machina/core:script_ops_op_lib",
        "//machina/distribute/experimental/rpc/kernels:rpc_ops",
        "//machina/dtensor/cc:dtensor_device_cc",
        "//machina/dtensor/cc:tensor_layout",
        "//machina/lite/c:common",
        "//machina/lite/core/api",
        "//machina/lite/delegates/flex:delegate",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/kernels:reference_ops",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/lite/toco/logging:conversion_log_util",
        "//machina/lite/toco/logging:toco_conversion_log_proto_cc",
        "//machina/compiler/mlir/lite:model_flags_proto_cc",
        "//machina/compiler/mlir/lite:converter_flags_proto_cc",
        "//machina/compiler/mlir/lite:types_proto_cc",
        "//machina/lite/toco:model_flags_proto_cc",
        "//machina/lite/toco:toco_convert",
        "//machina/lite/toco:toco_flags_proto_cc",
        "//machina/lite/toco:toco_graphviz_dump_options",
        "//machina/lite/toco:toco_port",
        "//machina/lite/toco:toco_tooling",
        "//machina/lite/toco:tooling_util",
        "//machina/lite/toco:types_proto_cc",
        "//machina/lite:framework",
        "//machina/lite:shared_library",
        "//machina/lite:stateful_error_reporter",
        "//machina/lite:string_util",
        "//machina/lite:util",
        "//machina/python/grappler:cost_analyzer_lib",
        "//machina/tools/graph_transforms:transform_graph_lib",
    ] + select({
        "//machina/compiler/mlir/python:disable_mlir_config": [],
        "//conditions:default": [
            "//machina/compiler/mlir/python:mlir",
        ],
    }) + (tf_monitoring_python_deps() +
          tf_additional_plugin_deps() +
          tf_additional_profiler_deps()) + if_xla_available([
        "//machina/compiler/aot:tfcompile_lib",
    ]) + if_static(extra_deps = [
        "//machina/core/platform:tensor_float_32_utils",
        "//machina/core/platform:enable_tf2_utils",
    ]) + if_oss([
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "//machina/core/distributed_runtime/rpc:grpc_session",
    ]),
    soversion = VERSION,
    static_deps = PACKAGE_STATIC_DEPS,
    visibility = ["//visibility:public"],
    win_def_file = ":machina_filtered_def_file",
)

# To avoid duplication, check that the C++ or python library does not depend on
# the stream executor cuda plugins. Targets that want to use cuda APIs should
# instead depend on the dummy plugins in //machina/compiler/xla/tsl/platform/default/build_config
# and use header only targets.
# TODO(ddunleavy): This seems completely broken. :machina_cc depends on
# cuda_platform from tf_additional_binary_deps and this doesn't break.
check_deps(
    name = "cuda_plugins_check_deps",
    disallowed_deps = if_static(
        [],
        otherwise = [
            "@local_xla//xla/stream_executor/cuda:all_runtime",
            "@local_xla//xla/stream_executor/cuda:cuda_platform",
            "@local_xla//xla/stream_executor/cuda:cudnn_plugin",
            "@local_xla//xla/stream_executor/cuda:cufft_plugin",
            "@local_xla//xla/stream_executor:cuda_platform",
        ],
    ),
    deps = if_cuda([
        "//machina:machina_cc",
        "//machina/python:pywrap_machina_internal",
    ]),
)

# ** Targets for Windows build (start) **

# Build a shared library (DLL) by cc_binary from tf_custom_op_library_additional_deps_impl,
# it contains all object code from its dependencies.
# This target is only used for parsing the symbols to be exported in machina.dll.
# Do NOT depend on it.
tf_native_cc_binary(
    name = "tf_custom_op_library_additional_deps.dll",
    linkshared = 1,
    linkstatic = 1,
    deps = tf_custom_op_library_additional_deps_impl(),
)

# Get a DEF file generated by parsing all object files
# of tf_custom_op_library_additional_deps.so
filegroup(
    name = "machina_def_file",
    srcs = [":tf_custom_op_library_additional_deps.dll"],
    output_group = "def_file",
)

# Filter the DEF file to reduce the number of symbols to 64K or less.
# Note that we also write the name of the pyd file into DEF file so that
# the dynamic libraries of custom ops can find it at runtime.
genrule(
    name = "machina_filtered_def_file",
    srcs = [":machina_def_file"],
    outs = ["machina_filtered_def_file.def"],
    cmd = select({
        "//machina:windows": """
              $(location @local_config_def_file_filter//:def_file_filter) \\
              --input $(location :machina_def_file) \\
              --output $@
          """,
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    tools = ["@local_config_def_file_filter//:def_file_filter"],
    visibility = ["//visibility:public"],
)

# The interface library (machina.dll.if.lib) for linking machina DLL library (machina.dll) on Windows.
# To learn more about import library (called interface library in Bazel):
#     https://docs.microsoft.com/en-us/cpp/build/linking-an-executable-to-a-dll?view=vs-2017#linking-implicitly
filegroup(
    name = "get_machina_dll_import_lib",
    srcs = ["//machina:machina.dll"],
    output_group = "interface_library",
    visibility = ["//visibility:public"],
)

# Rename the import library for machina.dll from machina.dll.if.lib to machina.lib
genrule(
    name = "machina_dll_import_lib",
    srcs = [":get_machina_dll_import_lib"],
    outs = ["machina.lib"],
    cmd = select({
        "//machina:windows": "cp -f $< $@",
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    visibility = ["//visibility:public"],
)

# The interface library (machina_cc.dll.if.lib) for linking machina DLL library (machina_cc.dll) on Windows.
# To learn more about import library (called interface library in Bazel):
#     https://docs.microsoft.com/en-us/cpp/build/linking-an-executable-to-a-dll?view=vs-2017#linking-implicitly
filegroup(
    name = "get_machina_cc_dll_import_lib",
    srcs = ["//machina:machina_cc.dll"],
    output_group = "interface_library",
    visibility = ["//visibility:public"],
)

# Rename the import library for machina.dll from machina_cc.dll.if.lib to machina.lib
genrule(
    name = "machina_cc_dll_import_lib",
    srcs = [":get_machina_cc_dll_import_lib"],
    outs = ["machina_cc.lib"],
    cmd = select({
        "//machina:windows": "cp -f $< $@",
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    visibility = ["//visibility:public"],
)

# ** Targets for Windows build (end) **

exports_files(
    [
        "tf_version_script.lds",
        "tf_exported_symbols.lds",
        "tf_private_symbols.lds",
    ],
)

genrule(
    name = "install_headers",
    srcs = [
        "//machina/c:headers",
        "//machina/c/eager:headers",
        "//machina/cc:headers",
        "//machina/core:headers",
    ],
    outs = ["include"],
    cmd = """
    mkdir $@
    for f in $(SRCS); do
      d="$${f%/*}"
      d="$${d#bazel-out/*/genfiles/}"
      d="$${d#bazel-out/*/bin/}"

      if [[ $${d} == *local_config_* ]]; then
        continue
      fi

      if [[ $${d} == external* ]]; then
        extname="$${d#*external/}"
        extname="$${extname%%/*}"
        if [[ $${TF_SYSTEM_LIBS:-} == *$${extname}* ]]; then
          continue
        fi

        d="$${d#*external/farmhash_archive/src}"
        d="$${d#*external/$${extname}/}"
        d="$${d#_virtual_includes/*/}"
      fi

      mkdir -p "$@/$${d}"
      cp "$${f}" "$@/$${d}/"
    done
    """,
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

# copybara:comment_begin(oss-only)
copy_file(
    name = "root_init_gen",
    src = select({
        "api_version_2": "_api/v2/v2.py",
        "//conditions:default": "_api/v1/v1.py",
    }),
    out = "__init__.py",
)

genrule(
    name = "virtual_root_init_gen",
    srcs = select({
        "api_version_2": [":virtual_root_template_v2.__init__.py"],
        "//conditions:default": [":virtual_root_template_v1.__init__.py"],
    }),
    outs = ["virtual_root.__init__.py"],
    cmd = "cp $(SRCS) $(OUTS)",
)

generate_apis(
    name = "tf_python_api_gen_v1",
    api_version = 1,
    compat_api_versions = [
        1,
        2,
    ],
    compat_init_templates = [
        "compat_template_v1.__init__.py",
        "compat_template.__init__.py",
    ],
    file_prefixes_to_strip = select({
        ":litert_prefixes": ["external/org_machina/"],
        "//conditions:default": [],
    }),
    output_dir = "_api/v1/",
    output_files = MACHINA_API_INIT_FILES_V1,
    output_package = "machina._api.v1",
    packages_to_ignore = ["machina.python.framework.test_ops"],
    root_file_name = "v1.py",
    root_init_template = "api_template_v1.__init__.py",
)

generate_apis(
    name = "tf_python_api_gen_v2",
    api_packages_file_name = "api_packages.txt",
    api_version = 2,
    compat_api_versions = [
        1,
        2,
    ],
    compat_init_templates = [
        "compat_template_v1.__init__.py",
        "compat_template.__init__.py",
    ],
    file_prefixes_to_strip = select({
        ":litert_prefixes": ["external/org_machina/"],
        "//conditions:default": [],
    }),
    output_dir = "_api/v2/",
    output_files = MACHINA_API_INIT_FILES_V2,
    output_package = "machina._api.v2",
    packages_to_ignore = select({
        ":disable_tf_lite_py": [
            "machina.python.framework.test_ops",
            "machina.lite",
        ],
        "//conditions:default": ["machina.python.framework.test_ops"],
    }),
    root_file_name = "v2.py",
    root_init_template = "api_template.__init__.py",
    visibility = ["//visibility:public"],
)

py_library(
    name = "machina_py",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":machina_py_no_contrib",
        "//machina/lite/python:analyzer",
        "//machina/lite/python:lite",
        "//machina/lite/python/authoring",
    ],
)

py_library(
    name = "machina_py_no_contrib",
    srcs = select({
        "api_version_2": [":tf_python_api_gen_v2"],
        "//conditions:default": [":tf_python_api_gen_v1"],
    }) + [
        ":root_init_gen",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//machina/lite/python:analyzer",
        "//machina/lite/python:lite",
        "//machina/lite/python/authoring",
        "//machina/python:no_contrib",
        "//machina/python/profiler:profiler_client",
        "@pypi_keras_nightly//:pkg",
        "@pypi_tb_nightly//:pkg",
    ],
)
# copybara:comment_end
