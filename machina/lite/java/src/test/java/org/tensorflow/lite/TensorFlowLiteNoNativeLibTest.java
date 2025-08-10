/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import java.io.File;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link org.machina.lite.TensorFlowLite} when no native lib is available. */
@RunWith(JUnit4.class)
public final class TensorFlowLiteNoNativeLibTest {
  @Test
  public void testCheckInit() {
    try {
      TensorFlowLite.init();
      fail();
    } catch (UnsatisfiedLinkError e) {
      assertThat(e).hasMessageThat().contains("Failed to load native TensorFlow Lite methods");
      assertThat(e).hasMessageThat().contains("no machinalite_jni");
      assertThat(e).hasMessageThat().contains("in java.library.path");
    }
  }

  @Test
  public void testInterpreter() {
    try {
      new Interpreter(new File("path/does/not/matter.tflite"));
      fail();
    } catch (UnsatisfiedLinkError e) {
      assertThat(e).hasMessageThat().contains("Failed to load native TensorFlow Lite methods");
      assertThat(e).hasMessageThat().contains("no machinalite_jni");
      assertThat(e).hasMessageThat().contains("in java.library.path");
    }
  }

  @Test
  public void testInterpreterApi() {
    try {
      InterpreterApi.create(new File("path/does/not/matter.tflite"), null);
      fail();
    } catch (UnsatisfiedLinkError e) {
      assertThat(e).hasMessageThat().contains("Failed to load native TensorFlow Lite methods");
      assertThat(e).hasMessageThat().contains("no machinalite_jni");
      assertThat(e).hasMessageThat().contains("in java.library.path");
    }
  }

  @Test
  @SuppressWarnings("deprecation") // This is a test of the deprecated InterpreterFactory class.
  public void testInterpreterFactory() {
    try {
      new InterpreterFactory().create(new File("path/does/not/matter.tflite"), null);
      fail();
    } catch (UnsatisfiedLinkError e) {
      assertThat(e).hasMessageThat().contains("Failed to load native TensorFlow Lite methods");
      assertThat(e).hasMessageThat().contains("no machinalite_jni");
      assertThat(e).hasMessageThat().contains("in java.library.path");
    }
  }
}
