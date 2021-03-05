#pragma once

#include "HMM.h"

extern "C" {
#include "GraphBLAS.h"
}

void check_for_error(const GrB_Info& info);

// Init GraphBLAS
// TODO: should be done exactly once
void launch_GraphBLAS();

// Finalize GraphBLAS
// TODO: should be done exactly once
void stop_GraphBLAS();

// Convert from GrB_Matrix to HMM::Prob_vec_t
// mat expected to be a column
HMM::Prob_vec_t GrB_Matrix_to_Prob_vec(GrB_Matrix mat);
