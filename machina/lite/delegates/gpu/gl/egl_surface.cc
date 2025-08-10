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

#include "machina/lite/delegates/gpu/gl/egl_surface.h"

#include <cstdint>
#include <utility>

#include "machina/lite/delegates/gpu/common/status.h"
#include "machina/lite/delegates/gpu/gl/gl_call.h"
#include "machina/lite/delegates/gpu/gl/gl_errors.h"

namespace tflite {
namespace gpu {
namespace gl {

EglSurface::EglSurface(EglSurface&& other)
    : surface_(other.surface_), display_(other.display_) {
  other.surface_ = EGL_NO_SURFACE;
}

EglSurface& EglSurface::operator=(EglSurface&& other) {
  if (this != &other) {
    display_ = other.display_;
    Invalidate();
    std::swap(surface_, other.surface_);
  }
  return *this;
}

void EglSurface::Invalidate() {
  if (surface_ != EGL_NO_SURFACE) {
    eglDestroySurface(display_, surface_);
    surface_ = EGL_NO_SURFACE;
  }
}

absl::Status CreatePbufferRGBSurface(EGLConfig config, EGLDisplay display,
                                     uint32_t height, uint32_t width,
                                     EglSurface* egl_surface) {
  const EGLint pbuffer_attributes[] = {EGL_WIDTH,
                                       static_cast<EGLint>(width),
                                       EGL_HEIGHT,
                                       static_cast<EGLint>(height),
                                       EGL_TEXTURE_FORMAT,
                                       EGL_TEXTURE_RGB,
                                       EGL_TEXTURE_TARGET,
                                       EGL_TEXTURE_2D,
                                       EGL_NONE};
  EGLSurface surface =
      eglCreatePbufferSurface(display, config, pbuffer_attributes);
  RETURN_IF_ERROR(GetOpenGlErrors());
  if (surface == EGL_NO_SURFACE) {
    return absl::InternalError(
        "No EGL error, but eglCreatePbufferSurface failed");
  }
  *egl_surface = EglSurface(surface, display);
  return absl::OkStatus();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
