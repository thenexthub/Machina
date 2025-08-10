/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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
#ifndef MACHINA_LITE_DELEGATES_COREML_BUILDERS_OP_VALIDATOR_H_
#define MACHINA_LITE_DELEGATES_COREML_BUILDERS_OP_VALIDATOR_H_

#include "machina/lite/core/c/builtin_op_data.h"

namespace tflite {
namespace delegates {
namespace coreml {
// Follow the ordering of TfLiteBuiltinOperator enum.
bool IsConcatenationOpSupported(const TfLiteRegistration* registration,
                                const TfLiteNode* node, TfLiteContext* context);
bool IsConvolutionOpSupported(const TfLiteRegistration* registration,
                              const TfLiteNode* node, TfLiteContext* context);
bool IsDepthwiseConvolutionOpSupported(const TfLiteRegistration* registration,
                                       const TfLiteNode* node,
                                       TfLiteContext* context);
bool IsFullyConnectedOpSupported(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context);
bool IsMeanOpSupported(const TfLiteRegistration* registration,
                       const TfLiteNode* node, TfLiteContext* context);
bool IsMirrorPadOpSupported(const TfLiteRegistration* registration,
                            const TfLiteNode* node, TfLiteContext* context);
bool IsPadOpSupported(const TfLiteRegistration* registration,
                      const TfLiteNode* node, TfLiteContext* context);
bool IsReshapeOpSupported(const TfLiteRegistration* registration,
                          const TfLiteNode* node, TfLiteContext* context,
                          int coreml_version);
bool IsResizeBilinearOpSupported(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context);
bool IsTransposeConvolutionOpSupported(const TfLiteRegistration* registration,
                                       const TfLiteNode* node,
                                       TfLiteContext* context);
}  // namespace coreml
}  // namespace delegates
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_COREML_BUILDERS_OP_VALIDATOR_H_
