integration_tests_seanet_quantize_GENERATOR_INPUTS := \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/quantize/quantize0.tflite \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/quantize/quantize1.tflite \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/quantize/quantize0_input0_int32.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/quantize/quantize0_golden_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/quantize/quantize1_input0_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/quantize/quantize1_golden_int32.csv

integration_tests_seanet_quantize_SRCS := \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/quantize/integration_tests.cc \
$(MACHINA_ROOT)python/tflite_micro/python_ops_resolver.cc \

integration_tests_seanet_quantize_HDR := \
$(MACHINA_ROOT)python/tflite_micro/python_ops_resolver.h \

$(eval $(call microlite_test,integration_tests_seanet_quantize_test,\
$(integration_tests_seanet_quantize_SRCS),$(integration_tests_seanet_quantize_HDR),$(integration_tests_seanet_quantize_GENERATOR_INPUTS)))
