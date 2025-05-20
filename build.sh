#!/bin/bash

rm -rf build
mkdir -p build
# cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake -S . -B build
make -C build VERBOSE=1
