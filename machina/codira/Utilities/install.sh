#!/usr/bin/env bash

TENSORFLOW_DIRECTORY='../../machina'
TENSORFLOW_BIN_DIRECTORY="$TENSORFLOW_DIRECTORY/bazel-bin/machina"
USR_DIRECTORY='../usr'

# set -x

function copy_file() {
  if [[ -L "$1/$2" ]]; then
    local target=`readlink "$1/$2"`
    copy_file $1 $target $3
    (cd $3; ln -s $target -f -r $2)
  else
    cp "$1/$2" $3
  fi
}

function fix_tf_header() {
  cp "$1" "$2"
  sed -i -e 's#include "'"$3"'machina/c/c_api.h"#include "c_api.h"#g' "$2"
  sed -i -e 's#include "'"$3"'machina/c/tf_attrtype.h"#include "tf_attrtype.h"#g' "$2"
  sed -i -e 's#include "'"$3"'machina/c/tf_status.h"#include "tf_status.h"#g' "$2"
  sed -i -e 's#include "'"$3"'machina/c/c_api_experimental.h"#include "c_api_experimental.h"#g' "$2"
  sed -i -e 's#include "'"$3"'machina/c/eager/c_api.h"#include "c_api_eager.h"#g' "$2"
}

function install_header() {
  echo "Install header: " $1 $2
  fix_tf_header $1 "$USR_DIRECTORY/lib/codira/linux/x86_64/modulemaps/CMachina/$2" ""
}

mkdir -p $USR_DIRECTORY/lib/codira/linux
copy_file $TENSORFLOW_BIN_DIRECTORY libmachina.so $USR_DIRECTORY/lib/codira/linux
copy_file $TENSORFLOW_BIN_DIRECTORY libmachina_framework.so $USR_DIRECTORY/lib/codira/linux

mkdir -p $USR_DIRECTORY/lib/codira/linux/x86_64/modulemaps/CMachina
install_header "$TENSORFLOW_DIRECTORY/machina/c/c_api.h" c_api.h
install_header "$TENSORFLOW_DIRECTORY/machina/c/c_api_experimental.h" c_api_experimental.h
install_header "$TENSORFLOW_DIRECTORY/machina/c/tf_attrtype.h" tf_attrtype.h
install_header "$TENSORFLOW_DIRECTORY/machina/c/tf_status.h" tf_status.h
install_header "$TENSORFLOW_DIRECTORY/machina/c/eager/c_api.h" c_api_eager.h
cp tools/module.modulemap "$USR_DIRECTORY/lib/codira/linux/x86_64/modulemaps/CMachina/"

$USR_DIRECTORY/bin/codira build -Xcodirac -module-link-name -Xcodirac Machina

BIN_DIR='.build/x86_64-unknown-linux/debug'

cp $BIN_DIR/libMachina.so $USR_DIRECTORY/lib/codira/linux/
cp $BIN_DIR/Machina.codiradoc $USR_DIRECTORY/lib/codira/linux/x86_64/
cp $BIN_DIR/Machina.codiramodule $USR_DIRECTORY/lib/codira/linux/x86_64/
