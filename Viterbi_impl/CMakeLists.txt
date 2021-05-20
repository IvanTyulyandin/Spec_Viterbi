cmake_minimum_required(VERSION 3.16)
project(Spec_Viterbi LANGUAGES CXX CUDA)

add_library(Viterbi_helper SHARED
    # Data storage and parsing
    HMM.h
    data_reader.h data_reader.cpp
    # Interfaces
    Viterbi_impl.h
    Viterbi_spec_impl.h
)


# GraphBLAS version
add_library(Viterbi_GraphBLAS STATIC
    GraphBLAS_helper.h GraphBLAS_helper.cpp
    GraphBLAS_impl.h GraphBLAS_impl.cpp
    GraphBLAS_spec_impl.h GraphBLAS_spec_impl.cpp
)
find_package(OpenMP REQUIRED)
target_link_libraries(Viterbi_GraphBLAS Viterbi_helper OpenMP::OpenMP_CXX graphblas)


# both for CUSP and cuASR versions
set(CMAKE_CUDA_FLAGS_RELEASE "--optimize 3 --generate-line-info -DNDEBUG")

# CUSP version
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


# cuASR version
add_library(cuASR INTERFACE)
target_include_directories(cuASR INTERFACE 
    ${CMAKE_CURRENT_LIST_DIR}/thirdparty/cuASR/include
    ${CMAKE_CURRENT_LIST_DIR}/thirdparty/cuASR/cutlass/include)

# Pascal
set(CUASR_CUDA_ARCHS "60 61")

add_library(Viterbi_cuASR SHARED
    cuASR_helper.h cuASR_helper.cu
    cuASR_impl.h cuASR_impl.cu
    cuASR_spec_impl.h cuASR_spec_impl.cu
)

set_target_properties(Viterbi_cuASR PROPERTIES CUDA_STANDARD 14)
set_target_properties(Viterbi_cuASR PROPERTIES CUDA_STANDARD_REQUIRED ON)
set_target_properties(Viterbi_cuASR PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(Viterbi_cuASR PUBLIC cudart cuASR Viterbi_helper)


# Result library
add_library(Spec_Viterbi INTERFACE)
target_link_libraries(Spec_Viterbi INTERFACE Viterbi_GraphBLAS Viterbi_CUSP Viterbi_cuASR)
