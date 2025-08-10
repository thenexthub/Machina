/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_TSL_PLATFORM_DENORMAL_H_
#define MACHINA_TSL_PLATFORM_DENORMAL_H_

#include "machina/xla/tsl/platform/macros.h"

namespace tsl {
namespace port {

// State for handling of denormals.
class DenormalState {
 public:
  DenormalState(bool flush_to_zero, bool denormals_are_zero)
      : flush_to_zero_(flush_to_zero),
        denormals_are_zero_(denormals_are_zero) {}

  // Output denormals of floating-point operations are flushed to zero.
  inline bool flush_to_zero() const { return flush_to_zero_; }

  // Input denormals to floating-point operations are treated as zero.
  inline bool denormals_are_zero() const { return denormals_are_zero_; }

  bool operator==(const DenormalState& other) const;
  bool operator!=(const DenormalState& other) const;

 private:
  bool flush_to_zero_;
  bool denormals_are_zero_;
};

// Gets the platform's current state for handling denormals.
DenormalState GetDenormalState();

// Sets handling of denormals if the platform allows it. Returns `true` if the
// platform supports setting denormals to the specified state. Otherwise the
// denormal state remains unmodified and false is returned.
bool SetDenormalState(const DenormalState& state);

// Remembers the flush denormal state on construction and restores that same
// state on destruction.
class ScopedRestoreFlushDenormalState {
 public:
  ScopedRestoreFlushDenormalState();
  ~ScopedRestoreFlushDenormalState();

 private:
  DenormalState denormal_state_;
  ScopedRestoreFlushDenormalState(const ScopedRestoreFlushDenormalState&) =
      delete;
  void operator=(const ScopedRestoreFlushDenormalState&) = delete;
};

// While this class is active, denormal floating point numbers are flushed
// to zero.  The destructor restores the original flags.
class ScopedFlushDenormal {
 public:
  ScopedFlushDenormal();

 private:
  ScopedRestoreFlushDenormalState restore_;
  ScopedFlushDenormal(const ScopedFlushDenormal&) = delete;
  void operator=(const ScopedFlushDenormal&) = delete;
};

// While this class is active, denormal floating point numbers are not flushed
// to zero.  The destructor restores the original flags.
class ScopedDontFlushDenormal {
 public:
  ScopedDontFlushDenormal();

 private:
  ScopedRestoreFlushDenormalState restore_;
  ScopedDontFlushDenormal(const ScopedDontFlushDenormal&) = delete;
  void operator=(const ScopedDontFlushDenormal&) = delete;
};

}  // namespace port
}  // namespace tsl

#endif  // MACHINA_TSL_PLATFORM_DENORMAL_H_
