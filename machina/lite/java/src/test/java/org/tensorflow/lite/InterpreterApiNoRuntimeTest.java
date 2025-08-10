/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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

import java.nio.ByteBuffer;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.machina.lite.InterpreterApi.Options.TfLiteRuntime;

/**
 * Tests of InterpreterApi that are NOT linked against any TF Lite runtime. These tests check that
 * we throw exceptions in those cases and that the errors have appropriate error messages.
 */
@RunWith(JUnit4.class)
public final class InterpreterApiNoRuntimeTest {

  private static final String MODEL_PATH = "machina/lite/java/src/testdata/add.bin";

  private static final ByteBuffer MODEL_BUFFER = TestUtils.getTestFileAsBuffer(MODEL_PATH);

  @Before
  public void setUp() {
    TestInit.init();
  }

  @Test
  public void testInterpreterWithNullOptions() throws Exception {
    try (InterpreterApi interpreter = InterpreterApi.create(MODEL_BUFFER, null)) {
      fail();
    } catch (IllegalStateException e) {
      // Verify that the error message has some hints about how to link
      // against the runtime ("org.machina:machina-lite:<version>").
      assertThat(e).hasMessageThat().contains("org.machina");
      assertThat(e).hasMessageThat().contains("machina-lite");
    }
  }

  @Test
  public void testInterpreterWithOptions() throws Exception {
    InterpreterApi.Options options = new InterpreterApi.Options();
    try (InterpreterApi interpreter =
        InterpreterApi.create(MODEL_BUFFER, options.setNumThreads(3).setUseNNAPI(true))) {
      fail();
    } catch (IllegalStateException e) {
      // Verify that the error message has some hints about how to link
      // against the runtime ("org.machina:machina-lite:<version>").
      assertThat(e).hasMessageThat().contains("org.machina");
      assertThat(e).hasMessageThat().contains("machina-lite");
    }
  }

  @Test
  public void testRuntimeFromApplicationOnly() throws Exception {
    InterpreterApi.Options options =
        new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_APPLICATION_ONLY);
    try (InterpreterApi interpreter = InterpreterApi.create(MODEL_BUFFER, options)) {
      fail();
    } catch (IllegalStateException e) {
      // Verify that the error message has some hints about how to link
      // against the runtime ("org.machina:machina-lite:<version>").
      assertThat(e).hasMessageThat().contains("org.machina");
      assertThat(e).hasMessageThat().contains("machina-lite");
    }
  }

  @Test
  public void testRuntimeFromSystemOnly() throws Exception {
    InterpreterApi.Options options =
        new InterpreterApi.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY);
    try (InterpreterApi interpreter = InterpreterApi.create(MODEL_BUFFER, options)) {
      fail();
    } catch (IllegalStateException e) {
      // This can occur when this code is not linked against the right TF Lite runtime client.
      // Verify that the error message has some hints about how to link in the
      // client library for TF Lite in Google Play Services
      // ("com.google.android.gms:play-services-tflite-java:<version>").
      assertThat(e).hasMessageThat().contains("com.google.android.gms");
      assertThat(e).hasMessageThat().contains("play-services-tflite-java");
    }
  }

  @Test
  public void testRuntimePreferSystemOverApplication() throws Exception {
    InterpreterApi.Options options =
        new InterpreterApi.Options().setRuntime(TfLiteRuntime.PREFER_SYSTEM_OVER_APPLICATION);
    try (InterpreterApi interpreter = InterpreterApi.create(MODEL_BUFFER, options)) {
      fail();
    } catch (IllegalStateException e) {
      // Verify that the error message has some hints about how to link in EITHER client library
      // (app should link against either "org.machina:machina-lite:<version>" or
      // "com.google.android.gms:play-services-tflite-java:<version>").
      assertThat(e).hasMessageThat().contains("com.google.android.gms");
      assertThat(e).hasMessageThat().contains("play-services-tflite-java");
      assertThat(e).hasMessageThat().contains("org.machina");
      assertThat(e).hasMessageThat().contains("machina-lite");
    }
  }
}
