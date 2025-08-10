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
/// \file
///
/// Defines the TFLITE_CONDITIONAL_NAMESPACE macro.
#ifndef MACHINA_LITE_SHIMS_NAMESPACE_H_
#define MACHINA_LITE_SHIMS_NAMESPACE_H_

// To avoid potential violation of the C++ "one definition rule" (ODR) for
// code which depends on TF Lite types that are conditionally defined
// (such as tflite::Interpreter, tflite::FlatBufferModel, TfLiteInterpreter
// and TfLiteModel), symbols that depend on such types should be defined
// in a (sub-)namespace whose name is also conditional.
#define TFLITE_CONDITIONAL_NAMESPACE regular_tflite

#endif  // MACHINA_LITE_NAMESPACE_H_
