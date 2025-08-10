/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

#ifndef MACHINA_TOOLS_ANDROID_INFERENCE_INTERFACE_JNI_RUN_STATS_JNI_H_
#define MACHINA_TOOLS_ANDROID_INFERENCE_INTERFACE_JNI_RUN_STATS_JNI_H_

#include <jni.h>

#include "machina/xla/tsl/util/stats_calculator.h"
#include "machina/core/util/stat_summarizer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define RUN_STATS_METHOD(name) \
  Java_org_machina_contrib_android_RunStats_##name

JNIEXPORT JNICALL jlong RUN_STATS_METHOD(allocate)(JNIEnv*, jclass);
JNIEXPORT JNICALL void RUN_STATS_METHOD(delete)(JNIEnv*, jclass, jlong);
JNIEXPORT JNICALL void RUN_STATS_METHOD(add)(JNIEnv*, jclass, jlong,
                                             jbyteArray);
JNIEXPORT JNICALL jstring RUN_STATS_METHOD(summary)(JNIEnv*, jclass, jlong);

#undef RUN_STATS_METHOD

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // MACHINA_TOOLS_ANDROID_INFERENCE_INTERFACE_JNI_RUN_STATS_JNI_H_
