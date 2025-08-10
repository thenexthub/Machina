/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#ifndef MACHINA_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_KEYPOINT_H_
#define MACHINA_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_KEYPOINT_H_

#include "machina/tools/android/test/jni/object_tracking/config.h"
#include "machina/tools/android/test/jni/object_tracking/geom.h"
#include "machina/tools/android/test/jni/object_tracking/image-inl.h"
#include "machina/tools/android/test/jni/object_tracking/image.h"
#include "machina/tools/android/test/jni/object_tracking/logging.h"
#include "machina/tools/android/test/jni/object_tracking/time_log.h"
#include "machina/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

// For keeping track of keypoints.
struct Keypoint {
  Keypoint() : pos_(0.0f, 0.0f), score_(0.0f), type_(0) {}
  Keypoint(const float x, const float y)
      : pos_(x, y), score_(0.0f), type_(0) {}

  Point2f pos_;
  float score_;
  uint8_t type_;
};

inline std::ostream& operator<<(std::ostream& stream, const Keypoint keypoint) {
  return stream << "[" << keypoint.pos_ << ", "
      << keypoint.score_ << ", " << keypoint.type_ << "]";
}

}  // namespace tf_tracking

#endif  // MACHINA_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_KEYPOINT_H_
