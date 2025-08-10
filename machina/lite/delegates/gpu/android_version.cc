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

#if defined(__ANDROID__)

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/system_properties.h>

extern "C" {

static int SdkVersion;
static int IsPreRelease;

static void readSystemProperties(void) {
  char buf[PROP_VALUE_MAX];

  if (__system_property_get("ro.build.version.sdk", buf) == 0) {
    // When the system property doesn't exist, defaults to future API level.
    SdkVersion = __ANDROID_API_FUTURE__;
  } else {
    SdkVersion = atoi(buf);  // NOLINT(runtime/deprecated_fn)
  }

  if (__system_property_get("ro.build.version.codename", buf) == 0) {
    IsPreRelease = 1;
  } else {
    IsPreRelease = strcmp(buf, "REL") != 0;
  }
  return;
}

int32_t __isOSVersionAtLeast(int32_t Major, int32_t Minor, int32_t Subminor) {
  (int32_t) Minor;
  (int32_t) Subminor;
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, readSystemProperties);

  return SdkVersion >= Major ||
         (IsPreRelease && Major == __ANDROID_API_FUTURE__);
}

}  // extern "C"

#endif  // defined(__ANDROID__)
