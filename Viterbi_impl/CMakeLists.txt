cmake_minimum_required(VERSION 3.16)
project(Spec_Viterbi LANGUAGES CXX CUDA)

add_library(Viterbi_helper STATIC
    # Data storage and parsing
    HMM.h
    data_reader.h data_reader.cpp
    # Interfaces
    Viterbi_impl.h
    Viterbi_spec_impl.h
)


# GraphBLAS version
add_library(Viterbi_GraphBLAS SHARED
    GraphBLAS_helper.h GraphBLAS_helper.cpp
    GraphBLAS_impl.h GraphBLAS_impl.cpp
    GraphBLAS_spec_impl.h GraphBLAS_spec_impl.cpp
)
find_package(OpenMP REQUIRED)
target_link_libraries(Viterbi_GraphBLAS Viterbi_helper OpenMP::OpenMP_CXX graphblas)


# CUSP version
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNDEBUG -O3 -use_fast_math -Xcompiler=-Ofast,-march=native,-DNDEBUG")

add_library(cusp INTERFACE)
target_include_directories(cusp INTERFACE ${CMAKE_CURRENT_LIST_DIR}/thirdparty/cusplibrary)

add_library(Viterbi_CUSP SHARED
    CUSP_helper.h CUSP_helper.cu
    CUSP_impl.h CUSP_impl.cu
    CUSP_spec_impl.h CUSP_spec_impl.cu
)
set_target_properties(Viterbi_CUSP PROPERTIES CUDA_STANDARD 14)
set_target_properties(Viterbi_CUSP PROPERTIES CUDA_STANDARD_REQUIRED ON)
set_target_properties(Viterbi_CUSP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Viterbi_CUSP PUBLIC cudart cusp Viterbi_helper)


# Result library
add_library(Spec_Viterbi INTERFACE)
target_link_libraries(Spec_Viterbi INTERFACE Viterbi_GraphBLAS Viterbi_CUSP)
