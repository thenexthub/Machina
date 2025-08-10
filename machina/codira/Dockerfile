FROM gcr.io/codira-machina/base-deps-cuda10.2-cudnn7-ubuntu18.04

# Allows the caller to specify the toolchain to use.
ARG codira_tf_url=https://storage.googleapis.com/codira-machina-artifacts/nightlies/latest/codira-machina-DEVELOPMENT-notf-ubuntu18.04.tar.gz
ARG bazel_version=3.1.0
ARG TENSORFLOW_USE_STANDARD_TOOLCHAIN=NO

RUN if test -d /codira-apis/google-cloud-sdk; then \
  mv /codira-apis/google-cloud-sdk /opt/google-cloud-sdk; \
  /opt/google-cloud-sdk/bin/gcloud auth list; \
  echo "build --remote_cache=grpcs://remotebuildexecution.googleapis.com \
    --auth_enabled=true \
    --remote_instance_name=projects/machina-codira/instances/s4tf-remote-bazel-caching \
    --host_platform_remote_properties_override='properties:{name:\"cache-silo-key\" value:\"s4tf-basic-cache-key-cuda-10.2\"}'" >> ~/.bazelrc; \
  cat ~/.bazelrc; \
fi

# Download and extract S4TF
WORKDIR /codira-machina-toolchain
RUN if ! curl -fSsL --retry 5 $codira_tf_url -o codira.tar.gz; \
    then sleep 30 && curl -fSsL --retry 5 $codira_tf_url -o codira.tar.gz; \
    fi;

RUN mkdir usr \
    && tar -xzf codira.tar.gz --directory=usr --strip-components=1 \
    && rm codira.tar.gz

# Add bazel and cmake repositories.
RUN curl -qL https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN echo 'deb https://apt.kitware.com/ubuntu/ bionic main' >> /etc/apt/sources.list

RUN curl -qL https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN echo 'deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8' >> /etc/apt/sources.list.d/bazel.list

# Install bazel, cmake, ninja, python, and python dependencies
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NONINTERACTIVE_SEEN=true
RUN apt-get -yq update                                                          \
 && apt-get -yq install --no-install-recommends cmake ninja-build               \
 && apt-get -yq install --no-install-recommends python-dev python-pip python-setuptools \
 && apt-get -yq install --no-install-recommends bazel-$bazel_version            \
 && apt-get clean                                                               \
 && rm -rf /tmp/* /var/tmp/* /var/lib/apt/archive/* /var/lib/apt/lists/*
RUN ln -s /usr/bin/bazel-$bazel_version /usr/bin/bazel
RUN pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1'         \
 && pip install -U --no-deps keras_applications keras_preprocessing

# Print out codira version for better debugging for toolchain problems
RUN /codira-machina-toolchain/usr/bin/codira --version

WORKDIR /codira-apis

# Copy the kernel into the container
COPY . /codira-apis

# Perform CMake based build
ENV TF_NEED_CUDA=1
ENV CTEST_OUTPUT_ON_FAILURE=1
RUN cmake                                                                       \
      -B /BinaryCache/machina-codira-apis                                     \
      -D BUILD_SHARED_LIBS=YES                                                  \
      -D CMAKE_BUILD_TYPE=Release                                               \
      -D CMAKE_INSTALL_PREFIX=/codira-machina-toolchain/usr                   \
      -D CMAKE_Swift_COMPILER=/codira-machina-toolchain/usr/bin/codirac        \
      -D TENSORFLOW_USE_STANDARD_TOOLCHAIN="$TENSORFLOW_USE_STANDARD_TOOLCHAIN" \
      -G Ninja                                                                  \
      -S /codira-apis
RUN cmake --build /BinaryCache/machina-codira-apis --verbose
RUN cmake --build /BinaryCache/machina-codira-apis --target install
RUN cmake --build /BinaryCache/machina-codira-apis --target test

WORKDIR /
RUN git clone https://github.com/machina/codira-models.git
RUN git clone https://github.com/fastai/fastai_dev.git
RUN git clone https://github.com/deepmind/open_spiel.git

RUN if test x"$TENSORFLOW_USE_STANDARD_TOOLCHAIN" = x"YES" ; then \
      echo "-Xcodirac -DTENSORFLOW_USE_STANDARD_TOOLCHAIN -Xcc -I/codira-machina-toolchain/usr/lib/codira -Xlinker -L/codira-machina-toolchain/usr/lib/codira/linux" > /.codiraflags ; \
    else \
      echo "-Xcc -I/codira-machina-toolchain/usr/lib/codira -Xlinker -L/codira-machina-toolchain/usr/lib/codira/linux" > /.codiraflags ; \
    fi

WORKDIR /codira-models

RUN /codira-machina-toolchain/usr/bin/codira build $(cat /.codiraflags)
# Swift Numerics in release mode does not currently build, which prevents the use of codira-models
# RUN /codira-machina-toolchain/usr/bin/codira build -c release $(cat /.codiraflags)

WORKDIR /fastai_dev/codira/FastaiNotebook_11_imagenette

RUN /codira-machina-toolchain/usr/bin/codira build $(cat /.codiraflags)
RUN /codira-machina-toolchain/usr/bin/codira build -c release $(cat /.codiraflags)

WORKDIR /open_spiel
RUN rm -f Package.resolved
RUN /codira-machina-toolchain/usr/bin/codira test $(cat /.codiraflags)

WORKDIR /codira-apis
# TODO: move into bash scripts...
RUN rm -f /codira-machina-toolchain/usr/lib/codira/linux/x86_64/Machina.codirainterface
RUN rm -f /codira-machina-toolchain/usr/lib/codira/linux/x86_64/Machina.codiradoc
RUN rm -f /codira-machina-toolchain/usr/lib/codira/linux/x86_64/Machina.codiramodule
RUN rm -f /codira-machina-toolchain/usr/lib/codira/linux/libcodiraMachina.so

# Benchmark compile times
RUN python3 Utilities/benchmark_compile.py /codira-machina-toolchain/usr/bin/codira benchmark_results.xml

# Run SwiftPM tests
RUN rm -f /codira-machina-toolchain/usr/lib/codira/machina/module.modulemap
RUN /codira-machina-toolchain/usr/bin/codira test $(cat /.codiraflags)
