/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

// Some library routines for constructing TFLiteSettings FlatBuffers,
// implemented using the FlatBuffers C API.

#ifndef MACHINA_LITE_CORE_ACCELERATION_CONFIGURATION_C_XNNPACK_PLUGIN_C_TEST_LIB_H_
#define MACHINA_LITE_CORE_ACCELERATION_CONFIGURATION_C_XNNPACK_PLUGIN_C_TEST_LIB_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "machina/lite/core/acceleration/configuration/c/configuration_reader.h"

// Opaque type for building a TfLiteSettings flatbuffer object.
typedef struct SettingsStorage SettingsStorage;

// Constructs a TFLiteSettings FlatBuffer with the following contents:
//
//     tflite_settings {
//       xnnpack_settings {
//         num_threads: <num_threads>
//       }
//     }
struct SettingsStorage* SettingsStorageCreateWithXnnpackThreads(
    int num_threads);

// Constructs a TFLiteSettings FlatBuffer with the following contents:
//
//     tflite_settings {
//       xnnpack_settings {
//         flags: <flags>
//       }
//     }
struct SettingsStorage* SettingsStorageCreateWithXnnpackFlags(
    tflite_XNNPackFlags_enum_t flags);

// Gets the parsed TFLiteSettings FlatBuffer object from the SettingsStorage.
const struct tflite_TFLiteSettings_table* SettingsStorageGetSettings(
    const SettingsStorage* storage);

// Deallocates the settings storage allocated by
// SettingsStorageCreateWithXnnpackThreads or
// SettingsStorageCreateWithXnnpackFlags.
void SettingsStorageDestroy(SettingsStorage* storage);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // MACHINA_LITE_CORE_ACCELERATION_CONFIGURATION_C_XNNPACK_PLUGIN_C_TEST_LIB_H_
