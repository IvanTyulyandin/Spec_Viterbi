cmake_minimum_required(VERSION 3.16)
project(tests)

find_program(MEMORYCHECK_COMMAND valgrind)
set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full")

# Parsing tests

add_executable(test_chmm_reader test_chmm_reader.cpp)
target_link_libraries(test_chmm_reader Spec_Viterbi)
add_test(test_chmm_reader test_chmm_reader)

add_executable(test_ess_reader test_ess_reader.cpp)
target_link_libraries(test_ess_reader Spec_Viterbi)
add_test(test_ess_reader test_ess_reader)

# GraphBLAS implementation tests

add_executable(test_GraphBLAS_impl test_GraphBLAS_impl.cpp test_helper.h)
target_link_libraries(test_GraphBLAS_impl Spec_Viterbi)
add_test(test_GraphBLAS_impl test_GraphBLAS_impl)

add_executable(test_GraphBLAS_spec_impl test_GraphBLAS_spec_impl.cpp test_helper.h)
target_link_libraries(test_GraphBLAS_spec_impl Spec_Viterbi)
add_test(test_GraphBLAS_spec_impl test_GraphBLAS_spec_impl)

# CUSP implementation tests

add_executable(test_CUSP_impl test_CUSP_impl.cpp test_helper.h)
target_link_libraries(test_CUSP_impl Spec_Viterbi)
add_test(test_CUSP_impl test_CUSP_impl)

add_executable(test_CUSP_spec_impl test_CUSP_spec_impl.cpp test_helper.h)
target_link_libraries(test_CUSP_spec_impl Spec_Viterbi)
add_test(test_CUSP_spec_impl test_CUSP_spec_impl)
