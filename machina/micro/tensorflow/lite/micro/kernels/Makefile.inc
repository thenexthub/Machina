###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Saturday, May 31, 2025.                                             #
#                                                                             #
#   Licensed under the Apache License, Version 2.0 (the "License");           #
#   you may not use this file except in compliance with the License.          #
#   You may obtain a copy of the License at:                                  #
#                                                                             #
#       http://www.apache.org/licenses/LICENSE-2.0                            #
#                                                                             #
#   Unless required by applicable law or agreed to in writing, software       #
#   distributed under the License is distributed on an "AS IS" BASIS,         #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
#   See the License for the specific language governing permissions and       #
#   limitations under the License.                                            #
#                                                                             #
#   Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,            #
#   Middletown, DE 19709, New Castle County, USA.                             #
#                                                                             #
###############################################################################=

# This file includes kernel test targets only.

# These tests needs additional dependencies beyond libmachina-microlite.a.
$(eval $(call microlite_test,kernel_detection_postprocess_test,\
  $(MACHINA_ROOT)machina/lite/micro/kernels/detection_postprocess_test.cc \
  $(MACHINA_ROOT)machina/lite/micro/kernels/detection_postprocess_flexbuffers_generated_data.cc,\
  $(MACHINA_ROOT)machina/lite/micro/kernels/detection_postprocess_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_circular_buffer_test,\
  $(MACHINA_ROOT)machina/lite/micro/kernels/circular_buffer_test.cc \
  $(MACHINA_ROOT)machina/lite/micro/kernels/circular_buffer_flexbuffers_generated_data.cc,\
  $(MACHINA_ROOT)machina/lite/micro/kernels/circular_buffer_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_conv_test,\
  $(MACHINA_ROOT)machina/lite/micro/kernels/conv_test.cc \
  $(MACHINA_ROOT)machina/lite/micro/kernels/conv_test_common.cc \
  $(MACHINA_ROOT)machina/lite/micro/kernels/testdata/conv_test_data.cc,\
  $(MACHINA_ROOT)machina/lite/micro/kernels/conv_test.h \
  $(MACHINA_ROOT)machina/lite/micro/kernels/testdata/conv_test_data.h))

$(eval $(call microlite_test,kernel_transpose_conv_test,\
  $(MACHINA_ROOT)machina/lite/micro/kernels/transpose_conv_test.cc \
  $(MACHINA_ROOT)machina/lite/micro/kernels/conv_test_common.cc,\
  $(MACHINA_ROOT)machina/lite/micro/kernels/conv_test.h))

$(eval $(call microlite_test,kernel_lstm_eval_test,\
  $(MACHINA_ROOT)machina/lite/micro/kernels/lstm_eval_test.cc \
  $(MACHINA_ROOT)machina/lite/micro/kernels/testdata/lstm_test_data.cc,\
  $(MACHINA_ROOT)machina/lite/micro/kernels/testdata/lstm_test_data.h))

$(eval $(call microlite_test,unidirectional_sequence_lstm_test,\
  $(MACHINA_ROOT)machina/lite/micro/kernels/unidirectional_sequence_lstm_test.cc \
  $(MACHINA_ROOT)machina/lite/micro/kernels/testdata/lstm_test_data.cc,\
  $(MACHINA_ROOT)machina/lite/micro/kernels/testdata/lstm_test_data.h))

$(eval $(call microlite_test,kernel_signal_delay_test,\
  $(MACHINA_ROOT)signal/micro/kernels/delay_test.cc \
  $(MACHINA_ROOT)signal/micro/kernels/delay_flexbuffers_generated_data.cc, \
  $(MACHINA_ROOT)signal/micro/kernels/delay_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_energy_test,\
  $(MACHINA_ROOT)signal/micro/kernels/energy_test.cc \
  $(MACHINA_ROOT)signal/micro/kernels/energy_flexbuffers_generated_data.cc, \
  $(MACHINA_ROOT)signal/micro/kernels/energy_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_fft_test,\
  $(MACHINA_ROOT)signal/micro/kernels/fft_test.cc \
  $(MACHINA_ROOT)signal/micro/kernels/fft_flexbuffers_generated_data.cc \
  $(MACHINA_ROOT)signal/testdata/fft_test_data.cc, \
  $(MACHINA_ROOT)signal/micro/kernels/fft_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_framer_test,\
  $(MACHINA_ROOT)signal/micro/kernels/framer_test.cc \
  $(MACHINA_ROOT)signal/micro/kernels/framer_flexbuffers_generated_data.cc, \
  $(MACHINA_ROOT)signal/micro/kernels/framer_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_filter_bank_test,\
  $(MACHINA_ROOT)signal/micro/kernels/filter_bank_test.cc \
  $(MACHINA_ROOT)signal/micro/kernels/filter_bank_flexbuffers_generated_data.cc, \
  $(MACHINA_ROOT)signal/micro/kernels/filter_bank_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_filter_bank_log_test,\
  $(MACHINA_ROOT)signal/micro/kernels/filter_bank_log_test.cc \
  $(MACHINA_ROOT)signal/micro/kernels/filter_bank_log_flexbuffers_generated_data.cc, \
  $(MACHINA_ROOT)signal/micro/kernels/filter_bank_log_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_filter_bank_square_root_test,\
  $(MACHINA_ROOT)signal/micro/kernels/filter_bank_square_root_test.cc))  

$(eval $(call microlite_test,kernel_signal_filter_bank_spectral_subtraction_test,\
  $(MACHINA_ROOT)signal/micro/kernels/filter_bank_spectral_subtraction_test.cc \
  $(MACHINA_ROOT)signal/micro/kernels/filter_bank_spectral_subtraction_flexbuffers_generated_data.cc, \
  $(MACHINA_ROOT)signal/micro/kernels/filter_bank_spectral_subtraction_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_overlap_add_test,\
  $(MACHINA_ROOT)signal/micro/kernels/overlap_add_test.cc \
  $(MACHINA_ROOT)signal/micro/kernels/overlap_add_flexbuffers_generated_data.cc, \
  $(MACHINA_ROOT)signal/micro/kernels/overlap_add_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_pcan_test,\
  $(MACHINA_ROOT)signal/micro/kernels/pcan_test.cc \
  $(MACHINA_ROOT)signal/micro/kernels/pcan_flexbuffers_generated_data.cc, \
  $(MACHINA_ROOT)signal/micro/kernels/pcan_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_stacker_test,\
  $(MACHINA_ROOT)signal/micro/kernels/stacker_test.cc \
  $(MACHINA_ROOT)signal/micro/kernels/stacker_flexbuffers_generated_data.cc, \
  $(MACHINA_ROOT)signal/micro/kernels/stacker_flexbuffers_generated_data.h))

$(eval $(call microlite_test,kernel_signal_window_test,\
  $(MACHINA_ROOT)signal/micro/kernels/window_test.cc \
  $(MACHINA_ROOT)signal/micro/kernels/window_flexbuffers_generated_data.cc, \
  $(MACHINA_ROOT)signal/micro/kernels/window_flexbuffers_generated_data.h))

# For kernel tests without extra dependencies (beyond libmachina-microlite.a),
# use simple for loop to generate their make targets in a common way.
MICROLITE_KERNEL_SIMPLE_TEST_SRCS := \
$(MACHINA_ROOT)machina/lite/micro/kernels/activations_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/add_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/add_n_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/arg_min_max_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/batch_matmul_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/batch_to_space_nd_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/broadcast_args_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/broadcast_to_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/cast_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/ceil_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/comparisons_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/concatenation_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/cumsum_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/depth_to_space_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/depthwise_conv_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/dequantize_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/div_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/elementwise_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/elu_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/embedding_lookup_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/exp_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/expand_dims_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/fill_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/floor_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/floor_div_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/floor_mod_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/fully_connected_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/gather_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/gather_nd_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/hard_swish_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/l2norm_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/l2_pool_2d_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/leaky_relu_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/logical_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/logistic_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/log_softmax_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/maximum_minimum_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/mirror_pad_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/mul_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/neg_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/pack_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/pad_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/pooling_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/prelu_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/quantization_util_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/quantize_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/reduce_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/reshape_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/resize_bilinear_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/resize_nearest_neighbor_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/reverse_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/round_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/select_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/shape_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/slice_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/softmax_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/space_to_batch_nd_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/space_to_depth_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/split_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/split_v_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/squared_difference_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/squeeze_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/strided_slice_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/sub_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/svdf_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/tanh_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/transpose_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/unpack_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/while_test.cc \
$(MACHINA_ROOT)machina/lite/micro/kernels/zeros_like_test.cc

ifeq ($(ENABLE_COMPRESSION), yes)
MICROLITE_KERNEL_SIMPLE_TEST_SRCS += $(MACHINA_ROOT)machina/lite/micro/kernels/decompress_test.cc
endif

# Generate simple kernel test targets in a common way
$(foreach TEST_TARGET,$(MICROLITE_KERNEL_SIMPLE_TEST_SRCS),\
$(eval $(call microlite_test,kernel_$(notdir $(basename $(TEST_TARGET))),$(TEST_TARGET))))
