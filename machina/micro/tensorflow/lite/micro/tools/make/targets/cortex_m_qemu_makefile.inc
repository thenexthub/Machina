###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, July 14, 2025.                                              #
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
###############################################################################

export PATH := $(DOWNLOADS_DIR)/gcc_embedded/bin/:$(PATH)
TARGET_TOOLCHAIN_PREFIX := arm-none-eabi-

DOWNLOAD_RESULT := $(shell $(MAKEFILE_DIR)/arm_gcc_download.sh ${DOWNLOADS_DIR} $(MACHINA_ROOT))
ifneq ($(DOWNLOAD_RESULT), SUCCESS)
  $(error Something went wrong with the GCC download: $(DOWNLOAD_RESULT))
endif

PLATFORM_FLAGS = \
  -DTF_LITE_MCU_DEBUG_LOG \
  -mcpu=$(TARGET_ARCH) \
  -mthumb \
  -mfloat-abi=soft \
  -mfpu=auto \
  -funsigned-char \
  -mlittle-endian \
  -fomit-frame-pointer

# Enable semihosting for QEMU to enable host system calls for debug_log and micro_time.
PLATFORM_FLAGS += --specs=rdimon.specs

CXXFLAGS += $(PLATFORM_FLAGS)
CCFLAGS += $(PLATFORM_FLAGS)

# TODO(b/158651472): Fix the memory_arena_threshold_test
EXCLUDED_TESTS := \
  $(MACHINA_ROOT)machina/lite/micro/memory_arena_threshold_test.cc

MICROLITE_TEST_SRCS := $(filter-out $(EXCLUDED_TESTS), $(MICROLITE_TEST_SRCS))

TEST_SCRIPT := $(MACHINA_ROOT)machina/lite/micro/testing/test_with_qemu.sh arm $(TARGET_ARCH)
