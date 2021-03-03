cmake_minimum_required(VERSION 3.18)
project(Spec_Viterbi)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-function -Wno-unused-variable)

# Build main part as Spec_Viterbi library
add_subdirectory(Viterbi_impl)

add_executable(Viterbi_alg main.cpp)
target_link_libraries(Viterbi_alg Spec_Viterbi)

# Build tests
include(CTest)
enable_testing()
add_subdirectory(tests)

# Copy dataset to a build folder
file(GLOB CHMM_DATA ${CMAKE_CURRENT_SOURCE_DIR}/chmm_files/*.chmm)
file(COPY ${CHMM_DATA} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/chmm_files/)

file(GLOB ESS_DATA ${CMAKE_CURRENT_SOURCE_DIR}/ess_files/*.ess)
file(COPY ${ESS_DATA} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/ess_files/)
