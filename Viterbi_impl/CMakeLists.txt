cmake_minimum_required(VERSION 3.16)
project(Spec_Viterbi)

# Build main part
set(LA_VITERBI_SRC
    # Data storage and parsing
    HMM.h
    data_reader.h data_reader.cpp
    # Interfaces
    Viterbi_impl.h
    Viterbi_spec_impl.h
    # GraphBLAS
    GraphBLAS_helper.h GraphBLAS_helper.cpp
    GraphBLAS_impl.h GraphBLAS_impl.cpp
    GraphBLAS_spec_impl.h GraphBLAS_spec_impl.cpp
    # CUSP
    CUSP_helper.h CUSP_helper.cpp
    CUSP_impl.h CUSP_impl.cpp
    CUSP_spec_impl.h CUSP_spec_impl.cpp)
add_library(Spec_Viterbi ${LA_VITERBI_SRC})

# Required by GraphBLAS
find_package(OpenMP REQUIRED)
target_link_libraries(Spec_Viterbi graphblas OpenMP::OpenMP_CXX)
