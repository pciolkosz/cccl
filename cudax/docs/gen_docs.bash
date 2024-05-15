#!/usr/bin/env bash

## This script just wraps launching a docs build within a container
## Tag is passed on as the first argument ${1}

set -e

SCRIPT_PATH=$(cd $(dirname ${0}); pwd -P)

cd $SCRIPT_PATH

## Clean image directory, without this any artifacts will prevent fetching
rm -rf img
mkdir -p img

./repo.sh docs || echo "!!! There were errors while generating"
