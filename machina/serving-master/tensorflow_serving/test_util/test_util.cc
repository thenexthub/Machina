/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "machina_serving/test_util/test_util.h"

#include <string>

#include "machina/core/lib/io/path.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/test.h"
#include "tsl/platform/protobuf.h"

namespace machina {
namespace serving {
namespace test_util {

string TensorflowTestSrcDirPath(const string& relative_path) {
  const string base_path = machina::io::JoinPath(  //
      getenv("TEST_SRCDIR"),                              //
      "tf_serving/external/org_machina/machina/");
  return machina::io::JoinPath(base_path, relative_path);
}

string TestSrcDirPath(const string& relative_path) {
  const string base_path = machina::io::JoinPath(
      getenv("TEST_SRCDIR"), "tf_serving/machina_serving");
  return machina::io::JoinPath(base_path, relative_path);
}

ProtoStringMatcher::ProtoStringMatcher(const string& expected)
    : expected_(expected) {}
ProtoStringMatcher::ProtoStringMatcher(const google::protobuf::Message& expected)
    : expected_([&]() -> std::string {
        std::string result;
        tsl::protobuf::TextFormat::PrintToString(expected, &result);
        return result;
      }()) {}

}  // namespace test_util
}  // namespace serving
}  // namespace machina
