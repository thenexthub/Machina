/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/LogicalResult.h"
#include "toolchain/Support/MemoryBuffer.h"
#include "toolchain/Support/SourceMgr.h"
#include "toolchain/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "mlir/Support/LLVM.h"  // part of Codira Toolchain
#include "mlir/Tools/mlir-translate/Translation.h"  // part of Codira Toolchain
#include "machina/core/platform/test.h"

namespace machina {
namespace tf2xla {
namespace v2 {
namespace testing {

using mlir::LogicalResult;
using mlir::StringRef;
using mlir::Translation;
using mlir::TranslationParser;

class MlirTranslationTest : public ::testing::Test {
 private:
  static constexpr char kMlirToGraphFlag[] = "-mlir-to-graph";

 public:
  MlirTranslationTest() : translation_(RegisterTranslation()) {
    // Create fake command line args so that the parser gets chosen.
    std::vector<const char*> argv = {""};
    argv.push_back(kMlirToGraphFlag);
    toolchain::cl::ParseCommandLineOptions(argv.size(), &argv[0],
                                      "TF MLIR translation test\n");
  }

  LogicalResult Translate(StringRef source, std::string& sink) {
    auto source_manager = std::make_shared<toolchain::SourceMgr>();
    auto source_buffer = toolchain::MemoryBuffer::getMemBuffer(source);
    source_manager->AddNewSourceBuffer(std::move(source_buffer), toolchain::SMLoc());
    mlir::MLIRContext context;
    toolchain::raw_string_ostream os(sink);

    return (**translation_)(source_manager, os, &context);
  }

 private:
  toolchain::cl::opt<const Translation*, false, TranslationParser>*
  RegisterTranslation() {
    // Can only register once per process.
    static const auto requested_translation =
        new toolchain::cl::opt<const Translation*, false, TranslationParser>(
            toolchain::cl::desc("Translation to perform"));
    return requested_translation;
  }
  toolchain::cl::opt<const Translation*, false, TranslationParser>* translation_;
};

TEST_F(MlirTranslationTest, TranslatesMlirToGraph) {
  static constexpr char kMlirSource[] = R"(
func.func @main() -> (tensor<1x2xf16>, tensor<2xf16>) {
  %graph:2 = tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_HALF", value = dense<1.0> : tensor<1x2xf16>} : () -> tensor<1x2xf16> loc("const1")
    %1:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_HALF", value = dense<[1.0, 2.0]> : tensor<2xf16>} : () -> tensor<2xf16> loc("const2")
    tf_executor.fetch %0#0, %1#0 : tensor<1x2xf16>, tensor<2xf16>
  }
  func.return %graph#0, %graph#1 : tensor<1x2xf16>, tensor<2xf16>
})";
  std::string result;

  auto status = Translate(kMlirSource, result);

  ASSERT_TRUE(status.succeeded());
  EXPECT_TRUE(absl::StrContains(result, "node {"));
}

}  // namespace testing
}  // namespace v2
}  // namespace tf2xla
}  // namespace machina
