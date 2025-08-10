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

#ifndef MACHINA_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_GL_UTILS_H_
#define MACHINA_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_GL_UTILS_H_

#include <GLES/gl.h>
#include <GLES/glext.h>

#include "machina/tools/android/test/jni/object_tracking/geom.h"

namespace tf_tracking {

// Draws a box at the given position.
inline static void DrawBox(const BoundingBox& bounding_box) {
  const GLfloat line[] = {
      bounding_box.left_, bounding_box.bottom_,
      bounding_box.left_, bounding_box.top_,
      bounding_box.left_, bounding_box.top_,
      bounding_box.right_, bounding_box.top_,
      bounding_box.right_, bounding_box.top_,
      bounding_box.right_, bounding_box.bottom_,
      bounding_box.right_, bounding_box.bottom_,
      bounding_box.left_, bounding_box.bottom_
  };

  glVertexPointer(2, GL_FLOAT, 0, line);
  glEnableClientState(GL_VERTEX_ARRAY);

  glDrawArrays(GL_LINES, 0, 8);
}


// Changes the coordinate system such that drawing to an arbitrary square in
// the world can thereafter be drawn to using coordinates 0 - 1.
inline static void MapWorldSquareToUnitSquare(const BoundingSquare& square) {
  glScalef(square.size_, square.size_, 1.0f);
  glTranslatef(square.x_ / square.size_, square.y_ / square.size_, 0.0f);
}

}  // namespace tf_tracking

#endif  // MACHINA_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_GL_UTILS_H_
