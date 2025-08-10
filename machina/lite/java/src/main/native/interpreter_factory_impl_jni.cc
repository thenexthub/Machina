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

#include <jni.h>
#include <stdio.h>

#include "machina/lite/c/c_api.h"
#include "machina/lite/java/src/main/native/jni_utils.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jstring JNICALL
Java_org_machina_lite_InterpreterFactoryImpl_nativeRuntimeVersion(
    JNIEnv* env, jclass /*clazz*/) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return nullptr;

  return env->NewStringUTF(TfLiteVersion());
}

JNIEXPORT jstring JNICALL
Java_org_machina_lite_InterpreterFactoryImpl_nativeSchemaVersion(
    JNIEnv* env, jclass /*clazz*/) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%d", TfLiteSchemaVersion());
  return env->NewStringUTF(buf);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
