cmake_minimum_required(VERSION 3.16)
project(benchmark)

add_library(bench_Viterbi 
    benchmark_helper.h
    bench_Viterbi.h bench_Viterbi_spec.h)
target_link_libraries(bench_Viterbi Spec_Viterbi stdc++fs)
