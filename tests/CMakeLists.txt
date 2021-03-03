cmake_minimum_required(VERSION 3.18)
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
