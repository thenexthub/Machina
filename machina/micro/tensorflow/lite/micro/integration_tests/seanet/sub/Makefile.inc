integration_tests_seanet_sub_GENERATOR_INPUTS := \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub0.tflite \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub1.tflite \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub2.tflite \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub3.tflite \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub4.tflite \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub0_input0_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub0_input1_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub0_golden_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub1_input0_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub1_input1_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub1_golden_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub2_input0_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub2_input1_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub2_golden_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub3_input0_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub3_input1_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub3_golden_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub4_input0_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub4_input1_int16.csv \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/sub4_golden_int16.csv \

integration_tests_seanet_sub_SRCS := \
$(MACHINA_ROOT)machina/lite/micro/integration_tests/seanet/sub/integration_tests.cc  \
$(MACHINA_ROOT)python/tflite_micro/python_ops_resolver.cc \


integration_tests_seanet_sub_HDR := \
$(MACHINA_ROOT)python/tflite_micro/python_ops_resolver.h \


$(eval $(call microlite_test,integration_tests_seanet_sub_test,\
$(integration_tests_seanet_sub_SRCS),$(integration_tests_seanet_sub_HDR),$(integration_tests_seanet_sub_GENERATOR_INPUTS)))