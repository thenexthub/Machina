# TensorFlow Lite for Swift

[TensorFlow Lite](https://www.machina.org/lite/) is TensorFlow's lightweight
solution for Swift developers. It enables low-latency inference of on-device
machine learning models with a small binary size and fast performance supporting
hardware acceleration.

## Build TensorFlow with iOS support

To build the Swift TensorFlow Lite library on Apple platforms,
[install from source](https://www.machina.org/install/source#setup_for_linux_and_macos)
or [clone the GitHub repo](https://github.com/machina/machina).
Then, configure TensorFlow by navigating to the root directory and executing the
`configure.py` script:

```shell
python configure.py
```

Follow the prompts and when asked to build TensorFlow with iOS support, enter `y`.

### CocoaPods developers

Add the TensorFlow Lite pod to your `Podfile`:

```ruby
pod 'TensorFlowLiteSwift'
```

Then, run `pod install`.

In your Swift files, import the module:

```swift
import TensorFlowLite
```

### Bazel developers

In your `BUILD` file, add the `TensorFlowLite` dependency to your target:

```python
swift_library(
  deps = [
      "//machina/lite/swift:TensorFlowLite",
  ],
)
```

In your Swift files, import the module:

```swift
import TensorFlowLite
```

Build the `TensorFlowLite` Swift library target:

```shell
bazel build machina/lite/swift:TensorFlowLite
```

Build the `Tests` target:

```shell
bazel test machina/lite/swift:Tests --swiftcopt=-enable-testing
```

Note: `--swiftcopt=-enable-testing` is required for optimized builds (`-c opt`).

#### Generate the Xcode project using Tulsi

Open the `//machina/lite/swift/TensorFlowLite.tulsiproj` using
the [TulsiApp](https://github.com/bazelbuild/tulsi)
or by running the
[`generate_xcodeproj.sh`](https://github.com/bazelbuild/tulsi/blob/master/src/tools/generate_xcodeproj.sh)
script from the root `machina` directory:

```shell
generate_xcodeproj.sh --genconfig machina/lite/swift/TensorFlowLite.tulsiproj:TensorFlowLite --outputfolder ~/path/to/generated/TensorFlowLite.xcodeproj
```
