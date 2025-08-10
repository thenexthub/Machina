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

import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.machina.lite.flex.FlexDelegate;
import org.machina.lite.nnapi.NnApiDelegate;

/**
 * Unit tests for {@link org.machina.lite.Interpreter} that validate execution with models that
 * have TensorFlow ops.
 */
@RunWith(JUnit4.class)
public final class InterpreterFlexTest {

  private static final ByteBuffer FLEX_MODEL_BUFFER =
      TestUtils.getTestFileAsBuffer("machina/lite/testdata/multi_add_flex.bin");

  /** Smoke test validating that flex model loading works when the flex delegate is used. */
  @Test
  public void testFlexModel() throws Exception {
    FlexDelegate delegate = new FlexDelegate();
    Interpreter.Options options = new Interpreter.Options().addDelegate(delegate);
    try (Interpreter interpreter = new Interpreter(FLEX_MODEL_BUFFER, options)) {
      testCommon(interpreter);
    } finally {
      delegate.close();
    }
  }

  /** Smoke test validating that flex model loading works when the flex delegate is linked. */
  @Test
  public void testFlexModelDelegateAutomaticallyApplied() throws Exception {
    try (Interpreter interpreter = new Interpreter(FLEX_MODEL_BUFFER)) {
      testCommon(interpreter);
    }
  }

  /** Smoke test validating that flex model loading works when the flex delegate is linked. */
  @Test
  public void testFlexModelDelegateAutomaticallyAppliedBeforeOtherDelegates() throws Exception {
    Interpreter.Options options = new Interpreter.Options();
    try (NnApiDelegate delegate = new NnApiDelegate();
        Interpreter interpreter =
            new Interpreter(FLEX_MODEL_BUFFER, options.addDelegate(delegate))) {
      testCommon(interpreter);
    }
  }

  private static void testCommon(Interpreter interpreter) {
    assertThat(interpreter.getInputTensorCount()).isEqualTo(4);
    assertThat(interpreter.getInputTensor(0).dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(interpreter.getInputTensor(1).dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(interpreter.getInputTensor(2).dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(interpreter.getInputTensor(3).dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(interpreter.getOutputTensorCount()).isEqualTo(2);
    assertThat(interpreter.getOutputTensor(0).dataType()).isEqualTo(DataType.FLOAT32);
    assertThat(interpreter.getOutputTensor(1).dataType()).isEqualTo(DataType.FLOAT32);

    float[] input1 = {1};
    float[] input2 = {2};
    float[] input3 = {3};
    float[] input4 = {5};
    Object[] inputs = new Object[] {input1, input2, input3, input4};

    float[] parsedOutput1 = new float[1];
    float[] parsedOutput2 = new float[1];
    Map<Integer, Object> outputs = new HashMap<>();
    outputs.put(0, parsedOutput1);
    outputs.put(1, parsedOutput2);

    interpreter.runForMultipleInputsOutputs(inputs, outputs);

    float[] expectedOutput1 = {6};
    float[] expectedOutput2 = {10};
    assertThat(parsedOutput1).usingTolerance(0.1f).containsExactly(expectedOutput1).inOrder();
    assertThat(parsedOutput2).usingTolerance(0.1f).containsExactly(expectedOutput2).inOrder();
  }

  static {
    FlexDelegate.initTensorFlowForTesting();
  }
}
