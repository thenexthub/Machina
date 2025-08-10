/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

package org.machina.lite.benchmark;

/** Helper class for running a native TensorFlow Lite benchmark. */
class BenchmarkModel {
  static {
    // Try loading flex first if available. If not load regular tflite shared library.
    try {
      System.loadLibrary("machinalite_benchmark_plus_flex");
    } catch (UnsatisfiedLinkError e) {
      System.loadLibrary("machinalite_benchmark");
    }
  }

  // Executes a standard TensorFlow Lite benchmark according to the provided args.
  //
  // Note that {@code args} will be split by the native execution code.
  public static void run(String args) {
    nativeRun(args);
  }

  private static native void nativeRun(String args);
}
