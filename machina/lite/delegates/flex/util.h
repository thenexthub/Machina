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
#ifndef MACHINA_LITE_DELEGATES_FLEX_UTIL_H_
#define MACHINA_LITE_DELEGATES_FLEX_UTIL_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "machina/c/c_api_internal.h"
#include "machina/c/tf_datatype.h"
#include "machina/core/framework/resource_handle.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/statusor.h"
#include "machina/lite/core/c/common.h"
#include "machina/lite/util.h"

namespace tflite {
namespace flex {

// Converts a machina:Status into a TfLiteStatus. If the original status
// represented an error, reports it using the given 'context'.
TfLiteStatus ConvertStatus(TfLiteContext* context, const absl::Status& status);

// Copies the given shape and type of the TensorFlow 'src' tensor into a TF Lite
// 'tensor'. Logs an error and returns kTfLiteError if the shape or type can't
// be converted.
TfLiteStatus CopyShapeAndType(TfLiteContext* context,
                              const machina::Tensor& src,
                              TfLiteTensor* tensor);

// Returns the TF C API Data type that corresponds to the given TfLiteType.
TF_DataType GetTensorFlowDataType(TfLiteType type);

// Returns the TfLiteType that corresponds to the given TF C API Data type.
TfLiteType GetTensorFlowLiteType(TF_DataType);

// Returns the TF type name that corresponds to the given TfLiteType.
const char* TfLiteTypeToTfTypeName(TfLiteType type);

// Creates a `machina::Tensor` from a TfLiteTensor for non-resource and
// non-variant type. Returns error status if the conversion fails.
absl::StatusOr<machina::Tensor> CreateTfTensorFromTfLiteTensor(
    const TfLiteTensor* tflite_tensor);

// Returns the encoded string name for a TF Lite resource variable tensor.
// This function will return a string in the format:
// tflite_resource_variable:resource_id.
std::string TfLiteResourceIdentifier(const TfLiteTensor* tensor);

// Parses out the resource ID from the given `resource_handle` and sets it
// to the corresponding TfLiteTensor. Returns true if succeed.
bool GetTfLiteResourceTensorFromResourceHandle(
    const machina::ResourceHandle& resource_handle, TfLiteTensor* tensor);

// We need a way to tell if we've stored a machina::Tensor* in a resource
// or if it's a standard kTfLiteResource tensor holding an integer. The proper
// solution would be some way to set the TfLiteTensor::type field to something
// unique for machina::Tensor* resources. We don't want to do that, so we use
// a hack instead: the `bytes` field of the tensor just needs to be big enough
// to hold a pointer, but it can be larger. To disambiguate between a pointer on
// a 32-bit machine and an int in a standard TFlite resource, we make the bytes
// field a fixed constant big enough for a pointer on any platform.
static constexpr int kTensorflowResourceTensorBytes = 8;

}  // namespace flex
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_FLEX_UTIL_H_
