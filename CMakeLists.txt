cmake_minimum_required(VERSION 3.16)
project(Spec_Viterbi)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -fPIC")


# Build main part as Spec_Viterbi library
add_subdirectory(Viterbi_impl)

add_subdirectory(benchmark)

add_executable(main main.cpp)
target_link_libraries(main bench_Viterbi)

# Build tests
include(CTest)
enable_testing()
add_subdirectory(tests)

# Copy dataset to a build folder
file(GLOB CHMM_DATA ${CMAKE_CURRENT_SOURCE_DIR}/chmm_files/*)
file(COPY ${CHMM_DATA} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/chmm_files/)

file(GLOB ESS_DATA ${CMAKE_CURRENT_SOURCE_DIR}/ess_files/*)
file(COPY ${ESS_DATA} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/ess_files/)
