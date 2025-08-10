/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef CUDA_CUDA_CONFIG_H_
#define CUDA_CUDA_CONFIG_H_

#define TF_CUDA_VERSION "%{cuda_version}"
#define TF_CUDART_VERSION "%{cudart_version}"
#define TF_CUPTI_VERSION "%{cupti_version}"
#define TF_CUBLAS_VERSION "%{cublas_version}"
#define TF_CUSOLVER_VERSION "%{cusolver_version}"
#define TF_CURAND_VERSION "%{curand_version}"
#define TF_CUFFT_VERSION "%{cufft_version}"
#define TF_CUSPARSE_VERSION "%{cusparse_version}"
#define TF_CUDNN_VERSION "%{cudnn_version}"

#define TF_CUDA_TOOLKIT_PATH "%{cuda_toolkit_path}"

#define TF_CUDA_COMPUTE_CAPABILITIES %{cuda_compute_capabilities}

#endif  // CUDA_CUDA_CONFIG_H_
