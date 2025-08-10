package(default_visibility = ["//visibility:public"])

# Please note that the output of these tools is unencumbered.
licenses(["restricted"])  # NCSA, GPLv3 (e.g. gold)

filegroup(
    name = "ar",
    srcs = ["toolchain/bin/toolchain-ar"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "as",
    srcs = ["toolchain/bin/toolchain-as"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "cpp",
    srcs = ["toolchain/bin/toolchain-cpp"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "dwp",
    srcs = ["toolchain/bin/toolchain-dwp"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "gcc",
    srcs = ["toolchain/bin/clang"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "gcov",
    srcs = ["toolchain/bin/toolchain-cov"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "ld",
    srcs = ["toolchain/bin/ld.lld"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "nm",
    srcs = ["toolchain/bin/toolchain-nm"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "objcopy",
    srcs = ["toolchain/bin/toolchain-objcopy"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "objdump",
    srcs = ["toolchain/bin/toolchain-objdump"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "profdata",
    srcs = ["toolchain/bin/toolchain-profdata"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "strip",
    srcs = ["sbin/strip"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "xray",
    srcs = ["toolchain/bin/toolchain-xray"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "includes",
    srcs = glob(["toolchain/lib/clang/6.0.0/include/**"]),
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "libraries",
    srcs = glob([
        "lib/*.*",
        "lib/clang/6.0.0/lib/linux/*.*",
    ]),
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "compiler_files",
    srcs = [
        ":as",
        ":gcc",
        ":includes",
    ],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "linker_files",
    srcs = [
        ":ar",
        ":ld",
        ":libraries",
    ],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "all_files",
    srcs = [
        ":compiler_files",
        ":dwp",
        ":gcov",
        ":linker_files",
        ":nm",
        ":objcopy",
        ":objdump",
        ":profdata",
        ":strip",
        ":xray",
    ],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "empty",
    srcs = [],  # bazel crashes without this
    output_licenses = ["unencumbered"],
)

cc_toolchain_suite(
    name = "clang6",
    toolchains = {
        "k8|clang6": ":clang6-k8",
    },
)

cc_toolchain(
    name = "clang6-k8",
    all_files = ":all_files",
    compiler_files = ":compiler_files",
    cpu = "k8",
    dwp_files = ":dwp",
    linker_files = ":linker_files",
    objcopy_files = ":objcopy",
    output_licenses = ["unencumbered"],
    strip_files = ":strip",
    supports_param_files = 1,
)
