# Spec Viterbi

This repository contains some experiments to
test specialized version vs non-specialized
version of the Viterbi algorithm, that is
expressed in terms of linear algebra.
Hidden Markov model (HMM) is supposed as a static
parameter, i.e. fixed.
The fact of the matrix multiplication
associativity is used to perform the
specialization.
The specialization is done as precalculations,
based on the info from the HMM.
For more info and details check out the code in
the `Viterbi_impl` folder.

## Build process

This section described what thirdparty software is required and how to build the project.

### Prerequisites

Built and installed [SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS "SuiteSparse:GraphBLAS repository").
On Linux systems, it may be required to properly
set LD_LIBRARY_PATH.

NVIDIA CUDA compatible device with driver and CUDA Toolkit installed.

### How to build

This repo has the dependency CUSP library as a git submodule.
Clone the repo as follows:

`git clone --recurse-submodules https://github.com/IvanTyulyandin/Spec_Viterbi`

There is a build script
`./build.sh`.

If you wish to run testing, you can run script
`./run_tests.sh`.
To check if memory leaks are present, you can
pass  
`-v` option to this script to run tests with Valgrind.
