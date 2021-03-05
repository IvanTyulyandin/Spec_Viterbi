cmake_minimum_required(VERSION 3.18)
project(Spec_Viterbi)

# Build main part
set(LA_VITERBI_SRC
    HMM.h
    data_reader.h data_reader.cpp
    Viterbi_impl.h
    Viterbi_spec_impl.h
    GraphBLAS_manager.h GraphBLAS_manager.cpp
    GraphBLAS_impl.h GraphBLAS_impl.cpp)
add_library(Spec_Viterbi ${LA_VITERBI_SRC})

# Required by GraphBLAS
find_package(OpenMP REQUIRED)
target_link_libraries(Spec_Viterbi graphblas OpenMP::OpenMP_CXX)
