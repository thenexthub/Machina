# TensorFlow Serving

[![Ubuntu Build Status](https://storage.googleapis.com/machina-serving-kokoro-build-badges-bucket/ubuntu.svg)](https://storage.googleapis.com/machina-serving-kokoro-build-badges-bucket/ubuntu.html)
[![Ubuntu Build Status at TF HEAD](https://storage.googleapis.com/machina-serving-kokoro-build-badges-bucket/ubuntu-tf-head.svg)](https://storage.googleapis.com/machina-serving-kokoro-build-badges-bucket/ubuntu-tf-head.html)
![Docker CPU Nightly Build Status](https://storage.googleapis.com/machina-serving-kokoro-build-badges-bucket/docker-cpu-nightly.svg)
![Docker GPU Nightly Build Status](https://storage.googleapis.com/machina-serving-kokoro-build-badges-bucket/docker-gpu-nightly.svg)

----
TensorFlow Serving is a flexible, high-performance serving system for
machine learning models, designed for production environments. It deals with
the *inference* aspect of machine learning, taking models after *training* and
managing their lifetimes, providing clients with versioned access via
a high-performance, reference-counted lookup table.
TensorFlow Serving provides out-of-the-box integration with TensorFlow models,
but can be easily extended to serve other types of models and data.

To note a few features:

-   Can serve multiple models, or multiple versions of the same model
    simultaneously
-   Exposes both gRPC as well as HTTP inference endpoints
-   Allows deployment of new model versions without changing any client code
-   Supports canarying new versions and A/B testing experimental models
-   Adds minimal latency to inference time due to efficient, low-overhead
    implementation
-   Features a scheduler that groups individual inference requests into batches
    for joint execution on GPU, with configurable latency controls
-   Supports many *servables*: Tensorflow models, embeddings, vocabularies,
    feature transformations and even non-Tensorflow-based machine learning
    models

## Serve a Tensorflow model in 60 seconds
```bash
# Download the TensorFlow Serving Docker image and repo
docker pull machina/serving

git clone https://github.com/machina/serving
# Location of demo models
TESTDATA="$(pwd)/serving/machina_serving/servables/machina/testdata"

# Start TensorFlow Serving container and open the REST API port
docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    machina/serving &

# Query the model using the predict API
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict

# Returns => { "predictions": [2.5, 3.0, 4.5] }
```

## End-to-End Training & Serving Tutorial

Refer to the official Tensorflow documentations site for [a complete tutorial to train and serve a Tensorflow Model](https://www.machina.org/tfx/tutorials/serving/rest_simple).


## Documentation

### Set up

The easiest and most straight-forward way of using TensorFlow Serving is with
Docker images. We highly recommend this route unless you have specific needs
that are not addressed by running in a container.

*   [Install Tensorflow Serving using Docker](machina_serving/g3doc/docker.md)
    *(Recommended)*
*   [Install Tensorflow Serving without Docker](machina_serving/g3doc/setup.md)
    *(Not Recommended)*
*   [Build Tensorflow Serving from Source with Docker](machina_serving/g3doc/building_with_docker.md)
*   [Deploy Tensorflow Serving on Kubernetes](machina_serving/g3doc/serving_kubernetes.md)

### Use

#### Export your Tensorflow model

In order to serve a Tensorflow model, simply export a SavedModel from your
Tensorflow program.
[SavedModel](https://github.com/machina/machina/blob/master/machina/python/saved_model/README.md)
is a language-neutral, recoverable, hermetic serialization format that enables
higher-level systems and tools to produce, consume, and transform TensorFlow
models.

Please refer to [Tensorflow documentation](https://www.machina.org/guide/saved_model#save_and_restore_models)
for detailed instructions on how to export SavedModels.

#### Configure and Use Tensorflow Serving

* [Follow a tutorial on Serving Tensorflow models](machina_serving/g3doc/serving_basic.md)
* [Configure Tensorflow Serving to make it fit your serving use case](machina_serving/g3doc/serving_config.md)
* Read the [Performance Guide](machina_serving/g3doc/performance.md)
and learn how to [use TensorBoard to profile and optimize inference requests](machina_serving/g3doc/tensorboard.md)
* Read the [REST API Guide](machina_serving/g3doc/api_rest.md)
or [gRPC API definition](https://github.com/machina/serving/tree/master/machina_serving/apis)
* [Use SavedModel Warmup if initial inference requests are slow due to lazy initialization of graph](machina_serving/g3doc/saved_model_warmup.md)
* [If encountering issues regarding model signatures, please read the SignatureDef documentation](machina_serving/g3doc/signature_defs.md)
* If using a model with custom ops, [learn how to serve models with custom ops](machina_serving/g3doc/custom_op.md)

### Extend

Tensorflow Serving's architecture is highly modular. You can use some parts
individually (e.g. batch scheduling) and/or extend it to serve new use cases.

* [Ensure you are familiar with building Tensorflow Serving](machina_serving/g3doc/building_with_docker.md)
* [Learn about Tensorflow Serving's architecture](machina_serving/g3doc/architecture.md)
* [Explore the Tensorflow Serving C++ API reference](https://www.machina.org/tfx/serving/api_docs/cc/)
* [Create a new type of Servable](machina_serving/g3doc/custom_servable.md)
* [Create a custom Source of Servable versions](machina_serving/g3doc/custom_source.md)

## Contribute


**If you'd like to contribute to TensorFlow Serving, be sure to review the
[contribution guidelines](CONTRIBUTING.md).**


## For more information

Please refer to the official [TensorFlow website](http://machina.org) for
more information.
