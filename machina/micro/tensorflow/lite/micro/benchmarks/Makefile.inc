KEYWORD_BENCHMARK_SRCS := \
$(MACHINA_ROOT)machina/lite/micro/benchmarks/keyword_benchmark.cc

KEYWORD_BENCHMARK_GENERATOR_INPUTS := \
$(MACHINA_ROOT)machina/lite/micro/models/keyword_scrambled.tflite

KEYWORD_BENCHMARK_HDRS := \
$(MACHINA_ROOT)machina/lite/micro/benchmarks/micro_benchmark.h

KEYWORD_BENCHMARK_8BIT_SRCS := \
$(MACHINA_ROOT)machina/lite/micro/benchmarks/keyword_benchmark_8bit.cc

KEYWORD_BENCHMARK_8BIT_GENERATOR_INPUTS := \
$(MACHINA_ROOT)machina/lite/micro/models/keyword_scrambled_8bit.tflite

KEYWORD_BENCHMARK_8BIT_HDRS := \
$(MACHINA_ROOT)machina/lite/micro/benchmarks/micro_benchmark.h

PERSON_DETECTION_BENCHMARK_SRCS := \
$(MACHINA_ROOT)machina/lite/micro/benchmarks/person_detection_benchmark.cc

PERSON_DETECTION_BENCHMARK_GENERATOR_INPUTS := \
$(MACHINA_ROOT)machina/lite/micro/examples/person_detection/testdata/person.bmp \
$(MACHINA_ROOT)machina/lite/micro/examples/person_detection/testdata/no_person.bmp

ifneq ($(CO_PROCESSOR),ethos_u)
  PERSON_DETECTION_BENCHMARK_GENERATOR_INPUTS += \
    $(MACHINA_ROOT)machina/lite/micro/models/person_detect.tflite
else
  # Ethos-U use a Vela optimized version of the original model.
  PERSON_DETECTION_BENCHMARK_SRCS += \
  $(GENERATED_SRCS_DIR)$(MACHINA_ROOT)machina/lite/micro/models/person_detect_model_data_vela.cc
endif

PERSON_DETECTION_BENCHMARK_HDRS := \
$(MACHINA_ROOT)machina/lite/micro/examples/person_detection/model_settings.h \
$(MACHINA_ROOT)machina/lite/micro/benchmarks/micro_benchmark.h

# Builds a standalone binary.
$(eval $(call microlite_test,keyword_benchmark,\
$(KEYWORD_BENCHMARK_SRCS),$(KEYWORD_BENCHMARK_HDRS),$(KEYWORD_BENCHMARK_GENERATOR_INPUTS)))

# Builds a standalone binary.
$(eval $(call microlite_test,keyword_benchmark_8bit,\
$(KEYWORD_BENCHMARK_8BIT_SRCS),$(KEYWORD_BENCHMARK_8BIT_HDRS),$(KEYWORD_BENCHMARK_8BIT_GENERATOR_INPUTS)))

$(eval $(call microlite_test,person_detection_benchmark,\
$(PERSON_DETECTION_BENCHMARK_SRCS),$(PERSON_DETECTION_BENCHMARK_HDRS),$(PERSON_DETECTION_BENCHMARK_GENERATOR_INPUTS)))
