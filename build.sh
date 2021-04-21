#!/bin/bash

mkdir -p cmake_build
cd cmake_build

CMAKE_ARGS='
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \'
case "$1" in
    -C)
        CMAKE_ARGS="$CMAKE_ARGS -DUSE_CUDA_CUSP=ON \\"
        ;;
esac

cmake $CMAKE_ARGS ..
make -j 5

cd ..
