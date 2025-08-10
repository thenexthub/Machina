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

package org.machina.lite.benchmark.delegateperformance;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;

/**
 * {@link Activity} class for Delegate Performance Latency Benchmark.
 *
 * <p>This Activity receives test arguments via a command line specified in an intent extra. It
 * passes the arguments to the {@link BenchmarkLatencyImpl} class to perform latency benchmark tests
 * via TFLite Benchmark Tool. Please check the test example in
 * machina/lite/tools/benchmark/experimental/delegate_performance/android/README.md.
 */
public class BenchmarkLatencyActivity extends Activity {

  private static final String TAG = "TfLiteBenchmarkLatency";
  private static final String TFLITE_SETTINGS_FILES_INTENT_KEY_0 = "--tflite_settings_files";
  private static final String ARGS_INTENT_KEY_0 = "--args";

  @Override
  public void onCreate(Bundle savedInstanceState) {
    Log.i(TAG, "Create benchmark latency activity.");
    super.onCreate(savedInstanceState);

    Intent intent = getIntent();
    Bundle bundle = intent.getExtras();
    String[] tfliteSettingsJsonFiles = bundle.getStringArray(TFLITE_SETTINGS_FILES_INTENT_KEY_0);
    String[] args = bundle.getStringArray(ARGS_INTENT_KEY_0);

    BenchmarkLatencyImpl impl =
        new BenchmarkLatencyImpl(getApplicationContext(), tfliteSettingsJsonFiles, args);
    if (impl.initialize()) {
      impl.benchmark();
    } else {
      Log.e(TAG, "Failed to initialize the latency benchmarking.");
    }
    finish();
  }
}
