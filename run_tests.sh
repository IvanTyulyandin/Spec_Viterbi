#!/bin/bash

cd cmake_build
case "$1" in
    -v)
        ctest --verbose -D ExperimentalMemCheck .
        exit 0
        ;;
    *) ctest .
        exit 0
        ;;
esac
cd ..
