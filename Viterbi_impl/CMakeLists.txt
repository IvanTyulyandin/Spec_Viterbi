cmake_minimum_required(VERSION 3.18)
project(src)

# Build main part
set(LA_VITERBI_SRC
    HMM.h
    data_reader.h data_reader.cpp)
add_library(Spec_Viterbi ${LA_VITERBI_SRC})
