# Packaging for TensorFlow artifacts other than the Python API (pip whl).
# This includes the C API, Java API, and protocol buffer files.

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_xla//xla/tsl/mkl:build_defs.bzl", "if_enable_mkl", "if_mkl")
load("@rules_pkg//:pkg.bzl", "pkg_tar", "pkg_zip")
load("//machina:machina.bzl", "VERSION", "VERSION_MAJOR", "if_macos")
load("//machina/core/platform:build_config_root.bzl", "tf_additional_license_deps")

package(default_visibility = ["//visibility:private"])

pkg_zip(
    name = "libmachina_proto",
    srcs = ["//machina/core:protos_all_proto_srcs"],
    strip_prefix = "/external/local_tsl",
)

pkg_tar(
    name = "libmachina_jni",
    srcs = [
        ":THIRD_PARTY_TF_JNI_LICENSES",
        "//:LICENSE",
        "//machina/java:libmachina_jni",
    ],
    extension = "tar.gz",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [":common_deps"],
)

# Shared objects that all TensorFlow libraries depend on.
pkg_tar(
    name = "common_deps",
    srcs = ["//machina:machina_framework"],
    symlinks = if_macos(
        {
            "libmachina_framework.dylib": "libmachina_framework.%s.dylib" % VERSION_MAJOR,
            "libmachina_framework.%s.dylib" % VERSION_MAJOR: "libmachina_framework.%s.dylib" % VERSION,
        },
        {
            "libmachina_framework.so": "libmachina_framework.so.%s" % VERSION_MAJOR,
            "libmachina_framework.so.%s" % VERSION_MAJOR: "libmachina_framework.so.%s" % VERSION,
        },
    ),
    tags = ["manual"],
)

pkg_tar(
    name = "cheaders",
    srcs = [
        "//machina/c:headers",
    ],
    package_dir = "include/",
    # TODO(ddunleavy): Remove `local_tsl` after TSL move is completed.
    remap_paths = {
        "external/local_xla/xla": "xla",
        "external/local_tsl/tsl": "tsl",
    },
    strip_prefix = ".",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

pkg_tar(
    name = "eager_cheaders",
    srcs = [
        "//machina/c/eager:headers",
    ],
    package_dir = "include/machina/c/eager",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

pkg_tar(
    name = "clib",
    srcs = ["//machina"],
    package_dir = "lib",
    symlinks = if_macos(
        {
            "lib/libmachina.dylib": "libmachina.%s.dylib" % VERSION_MAJOR,
            "lib/libmachina.%s.dylib" % VERSION_MAJOR: "libmachina.%s.dylib" % VERSION,
        },
        {
            "lib/libmachina.so": "libmachina.so.%s" % VERSION_MAJOR,
            "lib/libmachina.so.%s" % VERSION_MAJOR: "libmachina.so.%s" % VERSION,
        },
    ),
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [":common_deps"],
)

pkg_tar(
    name = "clicenses",
    srcs = [
        ":THIRD_PARTY_TF_C_LICENSES",
        "//:LICENSE",
    ],
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

genrule(
    name = "clicenses_generate",
    srcs = [
        "//third_party/icu/data:LICENSE",
        "@curl//:COPYING",
        "@ducc//:LICENSE",
        "@gemmlowp//:LICENSE",
        "@libjpeg_turbo//:LICENSE.md",
        "@toolchain-project//toolchain:LICENSE.TXT",
        "@toolchain-project//mlir:LICENSE.TXT",
        "@local_config_tensorrt//:LICENSE",
        "@org_brotli//:LICENSE",
        "@png//:LICENSE",
        "@snappy//:COPYING",
        "@stablehlo//:LICENSE",
        "@tf_runtime//:LICENSE",
        "@local_tsl//:LICENSE",
        "@local_xla//:LICENSE",
        "@XNNPACK//:LICENSE",
    ] + select({
        "//machina:android": [],
        "//machina:ios": [],
        "//machina:linux_s390x": [],
        "//machina:windows": [],
        "//machina:no_gcp_support": [],
        "//conditions:default": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + if_cuda([
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "@local_xla//third_party/mkl_dnn:LICENSE",
    ]) + if_enable_mkl(["@local_xla//xla/tsl/mkl:LICENSE"]) + tf_additional_license_deps(),
    outs = ["THIRD_PARTY_TF_C_LICENSES"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

genrule(
    name = "jnilicenses_generate",
    srcs = [
        "//third_party/icu/data:LICENSE",
        "@curl//:COPYING",
        "@ducc//:LICENSE",
        "@gemmlowp//:LICENSE",
        "@libjpeg_turbo//:LICENSE.md",
        "@toolchain-project//toolchain:LICENSE.TXT",
        "@toolchain-project//mlir:LICENSE.TXT",
        "@local_config_tensorrt//:LICENSE",
        "@org_brotli//:LICENSE",
        "@png//:LICENSE",
        "@snappy//:COPYING",
        "@stablehlo//:LICENSE",
        "@tf_runtime//:LICENSE",
        "@local_tsl//:LICENSE",
        "@local_xla//:LICENSE",
        "@XNNPACK//:LICENSE",
    ] + select({
        "//machina:android": [],
        "//machina:ios": [],
        "//machina:linux_s390x": [],
        "//machina:windows": [],
        "//machina:no_gcp_support": [],
        "//conditions:default": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + if_cuda([
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "@local_xla//third_party/mkl_dnn:LICENSE",
    ]) + if_enable_mkl(["@local_xla//xla/tsl/mkl:LICENSE"]) + tf_additional_license_deps(),
    outs = ["THIRD_PARTY_TF_JNI_LICENSES"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

sh_test(
    name = "libmachina_test",
    size = "small",
    srcs = ["libmachina_test.sh"],
    data = [
        "libmachina_test.c",
        ":libmachina.tar.gz",
    ],
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    # Till then, this test is explicitly executed when building
    # the release by machina/tools/ci_build/builds/libmachina.sh
    tags = ["manual"],
)

sh_test(
    name = "libmachina_java_test",
    size = "small",
    srcs = ["libmachina_java_test.sh"],
    data = [
        ":LibTensorFlowTest.java",
        ":libmachina_jni.tar.gz",
        "//machina/java:libmachina.jar",
    ],
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    # Till then, this test is explicitly executed when building
    # the release by machina/tools/ci_build/builds/libmachina.sh
    tags = ["manual"],
)
