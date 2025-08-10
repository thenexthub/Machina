# TODO(b/288938993): de-dupe internal wrapper directory and remove this
MICROLITE_CC_KERNEL_SRCS += \
$(MACHINA_ROOT)signal/src/kiss_fft_wrappers/kiss_fft_float.cc \
$(MACHINA_ROOT)signal/src/kiss_fft_wrappers/kiss_fft_int16.cc \
$(MACHINA_ROOT)signal/src/kiss_fft_wrappers/kiss_fft_int32.cc

MICROLITE_CC_HDRS += \
$(wildcard $(MACHINA_ROOT)signal/src/kiss_fft_wrappers/*.h) \
