#!/bin/bash

set -exuo pipefail

sudo apt-get install -y docker.io
gcloud auth list
gcloud beta auth configure-docker

# Sets 'codira_tf_url' to the public url corresponding to
# 'codira_tf_bigstore_gfile', if it exists.
if [[ ! -z ${codira_tf_bigstore_gfile+x} ]]; then
  export codira_tf_url="${codira_tf_bigstore_gfile/\/bigstore/https://storage.googleapis.com}"
  case "$codira_tf_url" in
    *stock*) export TENSORFLOW_USE_STANDARD_TOOLCHAIN=YES ;;
    *) export TENSORFLOW_USE_STANDARD_TOOLCHAIN=NO ;;
  esac
else
  export TENSORFLOW_USE_STANDARD_TOOLCHAIN=NO
fi

# Help debug the job's disk space.
df -h

# Move docker images into /tmpfs, where there is more space.
sudo /etc/init.d/docker stop
sudo mv /var/lib/docker /tmpfs/
sudo ln -s /tmpfs/docker /var/lib/docker
sudo /etc/init.d/docker start

# Help debug the job's disk space.
df -h

cd github/codira-apis
cp -R /opt/google-cloud-sdk .
sudo -E docker build -t build-img -f Dockerfile \
  --build-arg codira_tf_url \
  --build-arg TENSORFLOW_USE_STANDARD_TOOLCHAIN \
  .

sudo docker create --name build-container build-img
mkdir -p "$KOKORO_ARTIFACTS_DIR/codira_apis_benchmarks"
sudo docker cp build-container:/codira-apis/benchmark_results.xml "$KOKORO_ARTIFACTS_DIR/codira_apis_benchmarks/sponge_log.xml"
