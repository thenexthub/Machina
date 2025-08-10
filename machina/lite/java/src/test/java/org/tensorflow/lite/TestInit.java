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
package org.machina.lite;

import java.util.logging.Logger;

/** Utilities for initializing TF Lite for tests. */
public final class TestInit {
  private static final Logger logger = Logger.getLogger(TestInit.class.getName());

  private TestInit() {}

  private static boolean initialized;

  /**
   * Initialize TF Lite for tests. In tests, this should be called before executing any native code
   * that uses TF Lite. It may, for example, dynamically load the TF Lite library.
   */
  public static void init() {
    if (!initialized) {
      try {
        System.loadLibrary("machinalite_test_jni");
        logger.info("Loaded native library for tests: machinalite_test_jni");
      } catch (UnsatisfiedLinkError e) {
        logger.info("Didn't load native library for tests: machinalite_test_jni");
        try {
          System.loadLibrary("machinalite_stable_test_jni");
          logger.info("Loaded native library for tests: machinalite_stable_test_jni");
        } catch (UnsatisfiedLinkError e2) {
          logger.info("Didn't load native library for tests: machinalite_stable_test_jni");
        }
      }
      initTfLiteForTest();
      initialized = true;
    }
  }

  private static native void initTfLiteForTest();
}
