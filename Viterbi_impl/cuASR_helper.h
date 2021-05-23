#pragma once

#include "HMM.h"
#include <iostream>
#include <string>

namespace cuASR_helper {

class Dev_mat {
  public:
    Dev_mat() = default;
    Dev_mat(int rows, int cols);
    Dev_mat(const Dev_mat& rhs);
    Dev_mat(Dev_mat&& rhs);
    Dev_mat& operator=(const Dev_mat& rhs);
    Dev_mat& operator=(Dev_mat&& rhs);
    ~Dev_mat();

    HMM::Mod_prob_t* data;
    int rows;
    int cols;
    size_t bytes_size;
};

void validate_Dev_mat_ptr(const Dev_mat& mat, const std::string& msg);

void min_plus_Dev_mat_multiply(const Dev_mat& lhs, const Dev_mat& rhs, Dev_mat& res);

HMM::Mod_prob_vec_t Dev_mat_to_Prob_vec(const Dev_mat& mat);

void init_matrices_from_HMM(const HMM& hmm, Dev_mat& start_pr_dev, Dev_mat& transp_tr_dev,
                            std::vector<Dev_mat>& emit_mat_vec_dev);

} // namespace cuASR_helper
