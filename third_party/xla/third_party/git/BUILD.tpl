# Description:
# Exports generated files used to generate machina/core/util/version_info.cc

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(
    glob(["gen/*"]),
)
