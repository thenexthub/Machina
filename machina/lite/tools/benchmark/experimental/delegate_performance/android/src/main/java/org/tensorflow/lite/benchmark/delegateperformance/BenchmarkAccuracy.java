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
package org.machina.lite.benchmark.delegateperformance;

import android.content.Context;

/** Interface for Delegate Performance Accuracy Benchmark. */
public interface BenchmarkAccuracy {
  /**
   * Initializes and runs the accuracy benchmark.
   *
   * @param context the context to use for finding the test models and exporting reports
   * @param tfliteSettingsJsonFiles the list of paths to delegate JSON configurations
   */
  void benchmark(Context context, String[] tfliteSettingsJsonFiles);
}
