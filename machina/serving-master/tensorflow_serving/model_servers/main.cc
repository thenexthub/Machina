/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

// gRPC server implementation of
// machina_serving/apis/prediction_service.proto.
//
// It bring up a standard server to serve a single TensorFlow model using
// command line flags, or multiple models via config file.
//
// ModelServer prioritizes easy invocation over flexibility,
// and thus serves a statically configured set of models. New versions of these
// models will be loaded and managed over time using the
// AvailabilityPreservingPolicy at:
//     machina_serving/core/availability_preserving_policy.h.
// by AspiredVersionsManager at:
//     machina_serving/core/aspired_versions_manager.h
//
// ModelServer has inter-request batching support built-in, by using the
// BatchingSession at:
//     machina_serving/batching/batching_session.h
//
// To serve a single model, run with:
//     $path_to_binary/machina_model_server \
//     --model_base_path=[/tmp/my_model | gs://gcs_address]
// IMPORTANT: Be sure the base path excludes the version directory. For
// example for a model at /tmp/my_model/123, where 123 is the version, the base
// path is /tmp/my_model.
//
// To specify model name (default "default"): --model_name=my_name
// To specify port (default 8500): --port=my_port
// To enable batching (default disabled): --enable_batching
// To override the default batching parameters: --batching_parameters_file

#include <iostream>
#include <vector>

#include "machina/c/c_api.h"
#include "machina/compiler/jit/flags.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/init_main.h"
#include "machina/core/util/command_line_flags.h"
#include "machina_serving/model_servers/server.h"
#include "machina_serving/model_servers/version.h"

#if defined(LIBTPU_ON_GCE) || defined(PLATFORM_CLOUD_TPU)
#include "machina/core/protobuf/tpu/topology.pb.h"
#include "machina/core/tpu/tpu_global_init.h"

void InitializeTPU(machina::serving::main::Server::Options& server_options) {
  server_options.enforce_session_run_timeout = false;
  if (server_options.saved_model_tags.empty()) {
    server_options.saved_model_tags = "tpu,serve";
  }

  if (server_options.skip_initialize_tpu) {
    std::cout << "Skipping model server level Initializing TPU system.";
    return;
  }
  std::cout << "Initializing TPU system.";
  machina::tpu::TopologyProto tpu_topology;
  TF_QCHECK_OK(machina::InitializeTPUSystemGlobally(
      machina::Env::Default(), &tpu_topology))
      << "Failed to initialize TPU system.";
  std::cout << "Initialized TPU topology: " << tpu_topology.DebugString();
  server_options.num_request_iterations_for_warmup =
      tpu_topology.num_tpu_devices_per_task();
}
#endif

int main(int argc, char** argv) {
  machina::serving::main::Server::Options options;
  bool display_version = false;
  bool xla_cpu_compilation_enabled = false;
  bool xla_gpu_compilation_enabled = false;
  std::vector<machina::Flag> flag_list = {
      machina::Flag("port", &options.grpc_port,
                       "TCP port to listen on for gRPC/HTTP API. Disabled if "
                       "port set to zero."),
      machina::Flag("grpc_socket_path", &options.grpc_socket_path,
                       "If non-empty, listen to a UNIX socket for gRPC API "
                       "on the given path. Can be either relative or absolute "
                       "path."),
      machina::Flag("rest_api_port", &options.http_port,
                       "Port to listen on for HTTP/REST API. If set to zero "
                       "HTTP/REST API will not be exported. This port must be "
                       "different than the one specified in --port."),
      machina::Flag("rest_api_num_threads", &options.http_num_threads,
                       "Number of threads for HTTP/REST API processing. If not "
                       "set, will be auto set based on number of CPUs."),
      machina::Flag("rest_api_timeout_in_ms", &options.http_timeout_in_ms,
                       "Timeout for HTTP/REST API calls."),
      machina::Flag("rest_api_enable_cors_support",
                       &options.enable_cors_support,
                       "Enable CORS headers in response"),
      machina::Flag("enable_batching", &options.enable_batching,
                       "enable batching"),
      machina::Flag(
          "allow_version_labels_for_unavailable_models",
          &options.allow_version_labels_for_unavailable_models,
          "If true, allows assigning unused version labels to models that are "
          "not available yet."),
      machina::Flag("batching_parameters_file",
                       &options.batching_parameters_file,
                       "If non-empty, read an ascii BatchingParameters "
                       "protobuf from the supplied file name and use the "
                       "contained values instead of the defaults."),
      machina::Flag(
          "enable_per_model_batching_parameters",
          &options.enable_per_model_batching_params,
          "Enables model specific batching params like batch "
          "sizes, timeouts, batching feature flags to be read from "
          "`batching_params.pbtxt` file present in SavedModel dir "
          "of the model. Associated params in the global config "
          "from --batching_parameters_file are *ignored*. Only "
          "threadpool (name and size) related params are used from "
          "the global config, as this threadpool is shared across "
          "all the models that want to batch requests. This option "
          "is only applicable when --enable_batching flag is set."),
      machina::Flag("model_config_file", &options.model_config_file,
                       "If non-empty, read an ascii ModelServerConfig "
                       "protobuf from the supplied file name, and serve the "
                       "models in that file. This config file can be used to "
                       "specify multiple models to serve and other advanced "
                       "parameters including non-default version policy. (If "
                       "used, --model_name, --model_base_path are ignored.)"),
      machina::Flag("model_config_file_poll_wait_seconds",
                       &options.fs_model_config_poll_wait_seconds,
                       "Interval in seconds between each poll of the filesystem"
                       "for model_config_file. If unset or set to zero, "
                       "poll will be done exactly once and not periodically. "
                       "Setting this to negative is reserved for testing "
                       "purposes only."),
      machina::Flag("model_name", &options.model_name,
                       "name of model (ignored "
                       "if --model_config_file flag is set)"),
      machina::Flag("model_base_path", &options.model_base_path,
                       "path to export (ignored if --model_config_file flag "
                       "is set, otherwise required)"),
      machina::Flag("num_load_threads", &options.num_load_threads,
                       "The number of threads in the thread-pool used to load "
                       "servables. If set as 0, we don't use a thread-pool, "
                       "and servable loads are performed serially in the "
                       "manager's main work loop, may casue the Serving "
                       "request to be delayed. Default: 0"),
      machina::Flag("num_unload_threads", &options.num_unload_threads,
                       "The number of threads in the thread-pool used to "
                       "unload servables. If set as 0, we don't use a "
                       "thread-pool, and servable loads are performed serially "
                       "in the manager's main work loop, may casue the Serving "
                       "request to be delayed. Default: 0"),
      machina::Flag("max_num_load_retries", &options.max_num_load_retries,
                       "maximum number of times it retries loading a model "
                       "after the first failure, before giving up. "
                       "If set to 0, a load is attempted only once. "
                       "Default: 5"),
      machina::Flag("load_retry_interval_micros",
                       &options.load_retry_interval_micros,
                       "The interval, in microseconds, between each servable "
                       "load retry. If set negative, it doesn't wait. "
                       "Default: 1 minute"),
      machina::Flag("file_system_poll_wait_seconds",
                       &options.file_system_poll_wait_seconds,
                       "Interval in seconds between each poll of the "
                       "filesystem for new model version. If set to zero "
                       "poll will be exactly done once and not periodically. "
                       "Setting this to negative value will disable polling "
                       "entirely causing ModelServer to indefinitely wait for "
                       "a new model at startup. Negative values are reserved "
                       "for testing purposes only."),
      machina::Flag("flush_filesystem_caches",
                       &options.flush_filesystem_caches,
                       "If true (the default), filesystem caches will be "
                       "flushed after the initial load of all servables, and "
                       "after each subsequent individual servable reload (if "
                       "the number of load threads is 1). This reduces memory "
                       "consumption of the model server, at the potential cost "
                       "of cache misses if model files are accessed after "
                       "servables are loaded."),
      machina::Flag("machina_session_parallelism",
                       &options.machina_session_parallelism,
                       "Number of threads to use for running a "
                       "Tensorflow session. Auto-configured by default."
                       "Note that this option is ignored if "
                       "--platform_config_file is non-empty."),
      machina::Flag(
          "machina_session_config_file",
          &options.machina_session_config_file,
          "If non-empty, read an ascii TensorFlow Session "
          "ConfigProto protobuf from the supplied file name. Note, "
          "parts of the session config (threads, parallelism etc.) "
          "can be overridden if needed, via corresponding command "
          "line flags."),
      machina::Flag("machina_intra_op_parallelism",
                       &options.machina_intra_op_parallelism,
                       "Number of threads to use to parallelize the execution"
                       "of an individual op. Auto-configured by default."
                       "Note that this option is ignored if "
                       "--platform_config_file is non-empty."),
      machina::Flag("machina_inter_op_parallelism",
                       &options.machina_inter_op_parallelism,
                       "Controls the number of operators that can be executed "
                       "simultaneously. Auto-configured by default."
                       "Note that this option is ignored if "
                       "--platform_config_file is non-empty."),
      machina::Flag("use_alts_credentials", &options.use_alts_credentials,
                       "Use Google ALTS credentials"),
      machina::Flag(
          "ssl_config_file", &options.ssl_config_file,
          "If non-empty, read an ascii SSLConfig protobuf from "
          "the supplied file name and set up a secure gRPC channel"),
      machina::Flag("platform_config_file", &options.platform_config_file,
                       "If non-empty, read an ascii PlatformConfigMap protobuf "
                       "from the supplied file name, and use that platform "
                       "config instead of the Tensorflow platform. (If used, "
                       "--enable_batching is ignored.)"),
      machina::Flag(
          "per_process_gpu_memory_fraction",
          &options.per_process_gpu_memory_fraction,
          "Fraction that each process occupies of the GPU memory space "
          "the value is between 0.0 and 1.0 (with 0.0 as the default) "
          "If 1.0, the server will allocate all the memory when the server "
          "starts, If 0.0, Tensorflow will automatically select a value."),
      machina::Flag("saved_model_tags", &options.saved_model_tags,
                       "Comma-separated set of tags corresponding to the meta "
                       "graph def to load from SavedModel."),
      machina::Flag("grpc_channel_arguments",
                       &options.grpc_channel_arguments,
                       "A comma separated list of arguments to be passed to "
                       "the grpc server. (e.g. "
                       "grpc.max_connection_age_ms=2000)"),
      machina::Flag("grpc_max_threads", &options.grpc_max_threads,
                       "Max grpc server threads to handle grpc messages."),
      machina::Flag("enable_model_warmup", &options.enable_model_warmup,
                       "Enables model warmup, which triggers lazy "
                       "initializations (such as TF optimizations) at load "
                       "time, to reduce first request latency."),
      machina::Flag("num_request_iterations_for_warmup",
                       &options.num_request_iterations_for_warmup,
                       "Number of times a request is iterated during warmup "
                       "replay. This value is used only if > 0."),
      machina::Flag("version", &display_version, "Display version"),
      machina::Flag(
          "monitoring_config_file", &options.monitoring_config_file,
          "If non-empty, read an ascii MonitoringConfig protobuf from "
          "the supplied file name"),
      machina::Flag(
          "remove_unused_fields_from_bundle_metagraph",
          &options.remove_unused_fields_from_bundle_metagraph,
          "Removes unused fields from MetaGraphDef proto message to save "
          "memory."),
      machina::Flag("prefer_tflite_model", &options.prefer_tflite_model,
                       "EXPERIMENTAL; CAN BE REMOVED ANYTIME! "
                       "Prefer TensorFlow Lite model from `model.tflite` file "
                       "in SavedModel directory, instead of the TensorFlow "
                       "model from `saved_model.pb` file. "
                       "If no TensorFlow Lite model found, fallback to "
                       "TensorFlow model."),
      machina::Flag(
          "num_tflite_pools", &options.num_tflite_pools,
          "EXPERIMENTAL; CAN BE REMOVED ANYTIME! Number of TFLite interpreters "
          "in an interpreter pool of TfLiteSession. Typically there is one "
          "TfLiteSession for each TF Lite model that is loaded. If not "
          "set, will be auto set based on number of CPUs."),
      machina::Flag(
          "num_tflite_interpreters_per_pool",
          &options.num_tflite_interpreters_per_pool,
          "EXPERIMENTAL; CAN BE REMOVED ANYTIME! Number of TFLite interpreters "
          "in an interpreter pool of TfLiteSession. Typically there is one "
          "TfLiteSession for each TF Lite model that is loaded. If not "
          "set, will be 1."),
      machina::Flag(
          "enable_signature_method_name_check",
          &options.enable_signature_method_name_check,
          "Enable method_name check for SignatureDef. Disable this if serving "
          "native TF2 regression/classification models."),
      machina::Flag(
          "xla_cpu_compilation_enabled", &xla_cpu_compilation_enabled,
          "EXPERIMENTAL; CAN BE REMOVED ANYTIME! "
          "Enable XLA:CPU JIT (default is disabled). With XLA:CPU JIT "
          "disabled, models utilizing this feature will return bad Status "
          "on first compilation request."),
      machina::Flag(
          "xla_gpu_compilation_enabled", &xla_gpu_compilation_enabled,
          "EXPERIMENTAL; CAN BE REMOVED ANYTIME! "
          "Enable both XLA:CPU JIT and XLA:GPU JIT (default is disabled)."),
      machina::Flag("enable_profiler", &options.enable_profiler,
                       "Enable profiler service."),
      machina::Flag("thread_pool_factory_config_file",
                       &options.thread_pool_factory_config_file,
                       "If non-empty, read an ascii ThreadPoolConfig protobuf "
                       "from the supplied file name."),
      machina::Flag("mixed_precision", &options.mixed_precision,
                       "specify mixed_precision mode"),
      machina::Flag("skip_initialize_tpu", &options.skip_initialize_tpu,
                       "Whether to skip auto initializing TPU."),
      machina::Flag("enable_grpc_healthcheck_service",
                       &options.enable_grpc_healthcheck_service,
                       "Enable the standard gRPC healthcheck service."),
      machina::Flag(
          "enable_serialization_as_tensor_content",
          &options.enable_serialization_as_tensor_content,
          "Enable serialization of predict response as tensor content.")};

  const auto& usage = machina::Flags::Usage(argv[0], flag_list);
  if (!machina::Flags::Parse(&argc, argv, flag_list)) {
    std::cout << usage;
    return -1;
  }

  machina::port::InitMain(argv[0], &argc, &argv);
#if defined(LIBTPU_ON_GCE) || defined(PLATFORM_CLOUD_TPU)
  InitializeTPU(options);
#endif

  if (display_version) {
    std::cout << "TensorFlow ModelServer: " << TF_Serving_Version() << "\n"
              << "TensorFlow Library: " << TF_Version() << "\n";
    return 0;
  }

  if (argc != 1) {
    std::cout << "unknown argument: " << argv[1] << "\n" << usage;
  }

  if (!xla_cpu_compilation_enabled && !xla_gpu_compilation_enabled) {
    machina::DisableXlaCompilation();
  }

  machina::serving::main::Server server;
  const auto& status = server.BuildAndStart(options);
  if (!status.ok()) {
    std::cout << "Failed to start server. Error: " << status << "\n";
    return -1;
  }
  server.WaitForTermination();
  return 0;
}
