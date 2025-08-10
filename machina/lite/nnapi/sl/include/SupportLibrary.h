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

#ifndef MACHINA_LITE_NNAPI_SL_INCLUDE_SUPPORT_LIBRARY_H_
#define MACHINA_LITE_NNAPI_SL_INCLUDE_SUPPORT_LIBRARY_H_

#include <dlfcn.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <variant>

// Changed when importing from AOSP
#include "machina/lite/kernels/internal/compatibility.h"
#include "machina/lite/nnapi/NeuralNetworksTypes.h"
#include "machina/lite/nnapi/sl/public/NeuralNetworksSupportLibraryImpl.h"

namespace tflite {
namespace nnapi {

#ifndef __NNAPI_FL5_MIN_ANDROID_API__
#define __NNAPI_FL5_MIN_ANDROID_API__ __ANDROID_API_S__
#endif

/**
 * Helper struct, wraps different versions of NnApiSLDriverImpl.
 *
 * Owns the .so handle, and will close it in destructor.
 * Sets proper implStructFeatureLevel in constructor.
 *
 * There's expectation that for M>N, NnApiSLDriverImplFL(M) is
 * a strict superset of NnApiSLDriverImplFL(N), and *NnApiSLDriverImplFL(M) can
 * be reinterpret_cast to *NnApiSLDriverImplFL(N) safely.
 *
 * The base->implFeatureLevel is set to the actual Feature Level
 * implemented by the SLDriverImpl,
 */
struct NnApiSupportLibrary {
  NnApiSupportLibrary(const NnApiSLDriverImplFL5* impl, void* libHandle)
      : libHandle(libHandle), fl5(impl) {}
  // No need for ctor below since FL6&7 are typedefs of FL5
  // NnApiSupportLibrary(const NnApiSLDriverImplFL6& impl, void* libHandle):
  // impl(impl), NnApiSupportLibrary(const NnApiSLDriverImplFL7& impl, void*
  // libHandle): impl(impl), libHandle(libHandle) {}
  ~NnApiSupportLibrary() {
    if (libHandle != nullptr) {
      dlclose(libHandle);
      libHandle = nullptr;
    }
  }
  NnApiSupportLibrary(const NnApiSupportLibrary&) = delete;
  NnApiSupportLibrary& operator=(const NnApiSupportLibrary&) = delete;

  int64_t getFeatureLevel() const { return fl5->base.implFeatureLevel; }

  const NnApiSLDriverImplFL5* getFL5() const { return fl5; }
  const NnApiSLDriverImplFL6* getFL6() const {
    TFLITE_CHECK_GE(getFeatureLevel(), ANEURALNETWORKS_FEATURE_LEVEL_6);
    return reinterpret_cast<const NnApiSLDriverImplFL6*>(&fl5);
  }
  const NnApiSLDriverImplFL7* getFL7() const {
    TFLITE_CHECK_GE(getFeatureLevel(), ANEURALNETWORKS_FEATURE_LEVEL_7);
    return reinterpret_cast<const NnApiSLDriverImplFL6*>(&fl5);
  }

  void* libHandle = nullptr;
  const NnApiSLDriverImplFL5* fl5;
};

/**
 * Loads the NNAPI support library.
 * The NnApiSupportLibrary structure is filled with all the pointers. If one
 * function doesn't exist, a null pointer is stored.
 */
std::unique_ptr<const NnApiSupportLibrary> loadNnApiSupportLibrary(
    const std::string& libName);
std::unique_ptr<const NnApiSupportLibrary> loadNnApiSupportLibrary(
    void* libHandle);

}  // namespace nnapi
}  // namespace tflite

#endif  // MACHINA_LITE_NNAPI_SL_INCLUDE_SUPPORT_LIBRARY_H_
