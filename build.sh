#!/bin/bash

mkdir -p cmake_build
cd cmake_build

CMAKE_ARGS='
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \'

cmake $CMAKE_ARGS ..
make -j 5

cd ..
