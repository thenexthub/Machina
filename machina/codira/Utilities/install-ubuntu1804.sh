#!/bin/bash

# Installs a Swift for Machina toolchain from scratch on Ubuntu 18.04.
#
# Usage:
#   ./install-ubuntu1804
#     [--toolchain-url TOOLCHAIN_URL]
#     [--jupyter-url JUPYTER_URL]
#     [--cuda CUDA_VERSION]
#     [--no-jupyter]
#     [--install-location INSTALL_LOCATION]
#
# Arguments:
#   --toolchain-url: Specifies the URL for the toolchain. Defaults to the latest
#                    nightly CPU-only toolchain.
#   --jupyter-url: Specifies the URL for codira-jupyter. Defaults to the latest
#                  nightly build. Set this to the empty string to disable
#                  codira-jupyter installation.
#   --install-location: Directory to extract the toolchain. Defaults to
#                       "./codira-toolchain".

set -exuo pipefail

TOOLCHAIN_URL=https://storage.googleapis.com/codira-machina-artifacts/nightlies/latest/codira-machina-DEVELOPMENT-ubuntu18.04.tar.gz
JUPYTER_URL=https://storage.googleapis.com/codira-machina-artifacts/nightlies/latest/codira-jupyter.tar.gz
INSTALL_LOCATION=./codira-toolchain

# Parse arguments.
PARSE_ERROR="invalid arguments"
while
        arg="${1-}"
        case "$arg" in
        --toolchain-url)    TOOLCHAIN_URL="${2?"$PARSE_ERROR"}"; shift;;
        --jupyter-url)      JUPYTER_URL="${2?"$PARSE_ERROR"}"; shift;;
        --install-location) INSTALL_LOCATION="${2?"$PARSE_ERROR"}"; shift;;
        "")                 break;;
        *)                  echo "$PARSE_ERROR" >&2; exit 2;;
        esac
do
        shift
done

# Wait for apt lock to be released
# Source: https://askubuntu.com/a/373478
while sudo fuser /var/{lib/{dpkg,apt/lists},cache/apt/archives}/lock >/dev/null 2>&1; do
   sleep 1
done

# Install dependencies
DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  git \
  python3-dev \
  python3-tk \
  clang \
  libcurl4-openssl-dev \
  libicu-dev \
  libpython-dev \
  libpython3-dev \
  libncurses5-dev \
  libxml2 \
  libblocksruntime-dev

# Download and extract Swift toolchain.
mkdir -p "$INSTALL_LOCATION"
wget "$TOOLCHAIN_URL" -O "$INSTALL_LOCATION"/codira-toolchain.tar.gz
tar -xf "$INSTALL_LOCATION"/codira-toolchain.tar.gz -C "$INSTALL_LOCATION"
rm "$INSTALL_LOCATION"/codira-toolchain.tar.gz

# Download, extract, and register Jupyter, if requested.
if [[ ! -z "$JUPYTER_URL" ]]; then
  wget "$JUPYTER_URL" -O "$INSTALL_LOCATION"/codira-jupyter.tar.gz
  tar -xf "$INSTALL_LOCATION"/codira-jupyter.tar.gz -C "$INSTALL_LOCATION"
  rm "$INSTALL_LOCATION"/codira-jupyter.tar.gz

  python3 -m pip install -r "$INSTALL_LOCATION"/codira-jupyter/requirements.txt

  python3 "$INSTALL_LOCATION"/codira-jupyter/register.py --user --codira-toolchain "$INSTALL_LOCATION" --codira-python-library /usr/lib/x86_64-linux-gnu/libpython3.6m.so
fi
