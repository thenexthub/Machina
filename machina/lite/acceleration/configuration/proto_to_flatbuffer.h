/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#ifndef MACHINA_LITE_ACCELERATION_CONFIGURATION_PROTO_TO_FLATBUFFER_H_
#define MACHINA_LITE_ACCELERATION_CONFIGURATION_PROTO_TO_FLATBUFFER_H_

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "machina/lite/acceleration/configuration/configuration.pb.h"
#include "machina/lite/acceleration/configuration/configuration_generated.h"

namespace tflite {

// Converts the provided TFLiteSettings from proto to flatbuffer format.
// The returned TFLiteSettings pointer is only valid until either the
// FlatBufferBuilder is modified or when the FlatBufferBuilder's lifetime ends.
const TFLiteSettings* ConvertFromProto(
    const proto::TFLiteSettings& proto_settings,
    flatbuffers::FlatBufferBuilder* builder);

// Converts the provided ComputeSettings from proto to flatbuffer format.
// The returned ComputeSettings pointer is only valid until either the
// FlatBufferBuilder is modified or when the FlatBufferBuilder's lifetime ends.
const ComputeSettings* ConvertFromProto(
    const proto::ComputeSettings& proto_settings,
    flatbuffers::FlatBufferBuilder* builder);

// Converts the provided MiniBenchmarkSettings from proto to flatbuffer format.
// The returned MinibenchmarkSettings pointer is only valid until either the
// FlatBufferBuilder is modified or when the FlatBufferBuilder's lifetime ends.
const MinibenchmarkSettings* ConvertFromProto(
    const proto::MinibenchmarkSettings& proto_settings,
    flatbuffers::FlatBufferBuilder* builder);

}  // namespace tflite

#endif  // MACHINA_LITE_ACCELERATION_CONFIGURATION_PROTO_TO_FLATBUFFER_H_
