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
// LINT.IfChange

#ifndef MACHINA_CORE_UTIL_CTC_CTC_LOSS_UTIL_H_
#define MACHINA_CORE_UTIL_CTC_CTC_LOSS_UTIL_H_

#include <cmath>
#include <limits>

namespace machina {
namespace ctc {

template <class T>
constexpr T kLogZero() {
  return -std::numeric_limits<T>::infinity();  // NOLINT
}

// Add logarithmic probabilities using:
// ln(a + b) = ln(a) + ln(1 + exp(ln(b) - ln(a)))
// The two inputs are assumed to be log probabilities.
// (GravesTh) Eq. 7.18
template <typename T>
inline T LogSumExp(T log_prob_1, T log_prob_2) {
  // const T kLogZero = -std::numeric_limits<T>::infinity();
  // Always have 'b' be the smaller number to avoid the exponential from
  // blowing up.
  if (log_prob_1 == kLogZero<T>()) {
    return log_prob_2;
  } else if (log_prob_2 == kLogZero<T>()) {
    return log_prob_1;
  } else {
    return (log_prob_1 > log_prob_2)
               ? log_prob_1 + log1pf(expf(log_prob_2 - log_prob_1))
               : log_prob_2 + log1pf(expf(log_prob_1 - log_prob_2));
  }
}

}  // namespace ctc
}  // namespace machina

#endif  // MACHINA_CORE_UTIL_CTC_CTC_LOSS_UTIL_H_
// LINT.ThenChange(//machina/lite/kernels/ctc/ctc_loss_util.h)
