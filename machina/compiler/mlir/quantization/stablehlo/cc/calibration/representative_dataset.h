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
#ifndef MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_REPRESENTATIVE_DATASET_H_
#define MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_REPRESENTATIVE_DATASET_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "machina/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "machina/compiler/mlir/quantization/machina/quantization_options.pb.h"

namespace stablehlo::quantization {

// Translates a set of `RepresentativeDatsetConfig` to signature key ->
// `RepresentativeDatasetFile` mapping. This is useful when using
// `RepresentativeDatasetConfig`s at places that accept the legacy
// `RepresentativeDatasetFile` mapping.
// Returns a non-OK status when there is a duplicate signature key among
// `representative_dataset_configs`.
absl::StatusOr<absl::flat_hash_map<
    std::string, machina::quantization::RepresentativeDatasetFile>>
CreateRepresentativeDatasetFileMap(absl::Span<const RepresentativeDatasetConfig>
                                       representative_dataset_configs);

}  // namespace stablehlo::quantization

#endif  // MACHINA_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_REPRESENTATIVE_DATASET_H_
