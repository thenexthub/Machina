# Sample app for OVIC benchmarking.

load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

android_binary(
    name = "ovic_benchmarker_binary",
    srcs = [
        "OvicBenchmarkerActivity.java",
    ],
    assets = [
        "//machina/lite/java/ovic/src/testdata:coco_labels.txt",
        "//machina/lite/java/ovic/src/testdata:labels.txt",
        "//machina/lite/java/ovic/src/testdata:ovic_testdata",
    ],
    assets_dir = "",
    custom_package = "ovic.demo.app",
    manifest = "AndroidManifest.xml",
    multidex = "legacy",
    nocompress_extensions = [
        ".lite",
        ".tflite",
    ],
    resource_files = glob(["res/**"]),
    tags = ["manual"],
    deps = [
        # copybara:uncomment "//third_party/java/android/android_sdk_linux/extras/android/compatibility/multidex",
        "//machina/lite/java:machinalite",
        "//machina/lite/java/ovic:ovicbenchmarkerlib",
        "//machina/lite/java/ovic:ovicdetectionbenchmarkerlib",
    ],
)
