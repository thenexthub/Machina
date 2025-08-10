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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link org.machina.lite.DataType}. */
@RunWith(JUnit4.class)
public final class DataTypeTest {

  @Test
  public void testElemByteSize() {
    assertThat(DataType.FLOAT32.byteSize()).isEqualTo(4);
    assertThat(DataType.INT32.byteSize()).isEqualTo(4);
    assertThat(DataType.UINT8.byteSize()).isEqualTo(1);
    assertThat(DataType.INT64.byteSize()).isEqualTo(8);
    assertThat(DataType.STRING.byteSize()).isEqualTo(-1);
  }

  @Test
  public void testConversion() {
    for (DataType dataType : DataType.values()) {
      assertThat(DataTypeUtils.fromC(dataType.c())).isEqualTo(dataType);
    }
  }

  @Test
  public void testINT8AndUINT8() {
    assertThat(DataTypeUtils.toStringName(DataType.INT8)).isEqualTo("byte");
    assertThat(DataTypeUtils.toStringName(DataType.UINT8)).isEqualTo("byte");
    assertThat(DataTypeUtils.toStringName(DataType.INT8))
        .isEqualTo(DataTypeUtils.toStringName(DataType.UINT8));
  }
}
