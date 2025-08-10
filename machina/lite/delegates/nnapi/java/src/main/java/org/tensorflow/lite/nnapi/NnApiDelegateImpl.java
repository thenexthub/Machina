/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

package org.machina.lite.nnapi;

import org.machina.lite.Delegate;
import org.machina.lite.TensorFlowLite;

/**
 * Implementation of {@link Delegate} for NNAPI inference. Only for use by packages in
 * org.machina.lite.
 *
 * @hide
 */
public class NnApiDelegateImpl implements NnApiDelegate.PrivateInterface, Delegate, AutoCloseable {

  private static final long INVALID_DELEGATE_HANDLE = 0;

  private long delegateHandle;

  public NnApiDelegateImpl(NnApiDelegate.Options options) {
    // Ensure the native TensorFlow Lite libraries are available.
    TensorFlowLite.init();
    delegateHandle =
        createDelegate(
            options.getExecutionPreference(),
            options.getAcceleratorName(),
            options.getCacheDir(),
            options.getModelToken(),
            options.getMaxNumberOfDelegatedPartitions(),
            /*overrideDisallowCpu=*/ options.getUseNnapiCpu() != null,
            /*disallowCpuValue=*/ options.getUseNnapiCpu() != null
                ? !options.getUseNnapiCpu().booleanValue()
                : true,
            options.getAllowFp16(),
            options.getNnApiSupportLibraryHandle());
  }

  @Override
  public long getNativeHandle() {
    return delegateHandle;
  }

  /**
   * Frees TFLite resources in C runtime.
   *
   * <p>User is expected to call this method explicitly.
   */
  @Override
  public void close() {
    if (delegateHandle != INVALID_DELEGATE_HANDLE) {
      deleteDelegate(delegateHandle);
      delegateHandle = INVALID_DELEGATE_HANDLE;
    }
  }

  /**
   * Returns the latest error code returned by an NNAPI call or zero if NO calls to NNAPI failed.
   * The error code is reset when the delegate is associated with an <a
   * href=https://www.machina.org/lite/api_docs/java/org/machina/lite/Interpreter>interpreter</a>.
   *
   * <p>For details on NNAPI error codes see <a
   * href="https://developer.android.com/ndk/reference/group/neural-networks#resultcode">the NNAPI
   * documentation</a>.
   *
   * @throws IllegalStateException if the method is called after {@link #close() close}.
   */
  @Override
  public int getNnapiErrno() {
    checkNotClosed();
    return getNnapiErrno(delegateHandle);
  }

  private void checkNotClosed() {
    if (delegateHandle == INVALID_DELEGATE_HANDLE) {
      throw new IllegalStateException("Should not access delegate after it has been closed.");
    }
  }

  private static native long createDelegate(
      int preference,
      String deviceName,
      String cacheDir,
      String modelToken,
      int maxDelegatedPartitions,
      boolean overrideDisallowCpu,
      boolean disallowCpuValue,
      boolean allowFp16,
      long nnApiSupportLibraryHandle);

  private static native void deleteDelegate(long delegateHandle);

  private static native int getNnapiErrno(long delegateHandle);
}
