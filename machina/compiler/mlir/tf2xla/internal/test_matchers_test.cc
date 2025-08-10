/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, July 12, 2025.
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

#include "machina/compiler/mlir/tf2xla/internal/test_matchers.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"
#include "machina/compiler/tf2xla/xla_helpers.h"
#include "machina/xla/hlo/builder/xla_computation.h"
#include "machina/xla/service/hlo.pb.h"
#include "machina/xla/tsl/platform/statusor.h"
#include "machina/core/lib/monitoring/cell_reader.h"
#include "machina/core/lib/monitoring/counter.h"

namespace {
using ::machina::monitoring::testing::CellReader;
using ::testing::Not;

constexpr char kMetric[] = "/machina/metric";
auto* counter =
    machina::monitoring::Counter<1>::New(kMetric, "description", "status");
constexpr char kOkStatus[] = "ok";
const int kArbitraryIntResult = 37;

template <typename T>
tsl::StatusOr<T> success(T t) {
  return t;
}
absl::StatusOr<int> success() { return kArbitraryIntResult; }
template <typename T>
tsl::StatusOr<T> filtered(T t) {
  return tsl::StatusOr<T>(machina::CompileToHloGraphAnalysisFailedError());
}
absl::StatusOr<int> filtered() { return filtered(kArbitraryIntResult); }
absl::StatusOr<int> failed() {
  return absl::StatusOr<int>(absl::InternalError("fail"));
}

TEST(TestUtil, MatchesOk) { ASSERT_THAT(success(), IsOkOrFiltered()); }

TEST(TestUtil, DoesntMatchesFailure) {
  ASSERT_THAT(failed(), Not(IsOkOrFiltered()));
}

TEST(TestUtil, MatchesFiltered) { ASSERT_THAT(filtered(), IsOkOrFiltered()); }

TEST(TestUtil, IncrementsOk) {
  CellReader<int64_t> reader(kMetric);
  counter->GetCell(kOkStatus)->IncrementBy(1);

  ASSERT_THAT(success(), IncrementedOrFiltered(reader.Delta(kOkStatus), 1));
}

TEST(TestUtil, FilteredDoesntIncrementsOk) {
  CellReader<int64_t> reader(kMetric);

  ASSERT_THAT(filtered(), IncrementedOrFiltered(reader.Delta(kOkStatus), 1));
}

TEST(TestUtil, FailureDoesntMatchIncrement) {
  CellReader<int64_t> reader(kMetric);

  ASSERT_THAT(failed(), Not(IncrementedOrFiltered(reader.Delta(kOkStatus), 1)));
}

machina::XlaCompilationResult CreateXlaComputationResult(
    const char* hlo_name) {
  auto result = machina::XlaCompilationResult();
  xla::HloModuleProto hlo;
  hlo.set_name(hlo_name);
  result.computation = std::make_shared<xla::XlaComputation>(hlo);
  return result;
}

TEST(TestUtil, ComputationContainsOk) {
  constexpr char arbitrary_hlo[] = "arbitrary_hlo";
  auto result = CreateXlaComputationResult(arbitrary_hlo);

  ASSERT_THAT(success(result), ComputationProtoContains(arbitrary_hlo));
}

TEST(TestUtil, ComputationDoesNotContain) {
  constexpr char arbitrary_hlo[] = "arbitrary_hlo";
  constexpr char bad_hlo[] = "bad_hlo";
  auto result = CreateXlaComputationResult(arbitrary_hlo);

  ASSERT_THAT(success(result), Not(ComputationProtoContains(bad_hlo)));
}

TEST(TestUtil, ComputationDoesNotContainFiltered) {
  constexpr char arbitrary_hlo[] = "arbitrary_hlo";
  constexpr char bad_hlo[] = "bad_hlo";
  auto result = CreateXlaComputationResult(arbitrary_hlo);

  ASSERT_THAT(filtered(result), ComputationProtoContains(bad_hlo));
}

TEST(TestUtil, MlirModuleHas) {
  constexpr char arbirary_mlir[] = "arbirary_mlir";

  ASSERT_THAT(success(arbirary_mlir), HasMlirModuleWith(arbirary_mlir));
}

TEST(TestUtil, MlirModuleDoesNotHave) {
  constexpr char arbirary_mlir[] = "arbirary_mlir";
  constexpr char bad_mlir[] = "bad_mlir";

  ASSERT_THAT(success(arbirary_mlir), Not(HasMlirModuleWith(bad_mlir)));
}

TEST(TestUtil, MlirModuleDoesNotHaveFiltered) {
  constexpr char arbirary_mlir[] = "arbirary_mlir";
  constexpr char bad_mlir[] = "bad_mlir";

  ASSERT_THAT(filtered(arbirary_mlir), HasMlirModuleWith(bad_mlir));
}

}  // namespace
