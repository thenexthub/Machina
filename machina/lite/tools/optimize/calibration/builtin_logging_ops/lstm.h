/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#ifndef MACHINA_LITE_TOOLS_OPTIMIZE_CALIBRATION_BUILTIN_LOGGING_OPS_LSTM_H_
#define MACHINA_LITE_TOOLS_OPTIMIZE_CALIBRATION_BUILTIN_LOGGING_OPS_LSTM_H_

#include "machina/lite/core/api/error_reporter.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/tools/optimize/calibration/calibration_logger.h"

namespace tflite {
namespace optimize {
namespace calibration {
namespace builtin {

enum class LSTMType {
  kLSTM,
  kUnidirectionalSequenceLSTM,
};

TfLiteStatus lstm_logging_kernel(TfLiteContext* context,
                                 const int subgraph_index, TfLiteNode* node,
                                 Logger* logger, ErrorReporter* error_reporter);

TfLiteStatus unidirectional_sequence_lstm_logging_kernel(
    TfLiteContext* context, const int subgraph_index, TfLiteNode* node,
    Logger* logger, ErrorReporter* error_reporter);

}  // namespace builtin
}  // namespace calibration
}  // namespace optimize
}  // namespace tflite

#endif  // MACHINA_LITE_TOOLS_OPTIMIZE_CALIBRATION_BUILTIN_LOGGING_OPS_LSTM_H_
