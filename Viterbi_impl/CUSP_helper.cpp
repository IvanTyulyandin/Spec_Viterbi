#include "CUSP_helper.h"

#include <cusp/functional.h>
#include <cusp/multiply.h>
#include <limits>

namespace CUSP_helper {

constexpr auto inf = std::numeric_limits<HMM::Probability_t>::infinity();

void min_plus_Dev_mat_multiply(const Dev_mat& lhs, const Dev_mat& rhs, Dev_mat& res) {
    const auto init = cusp::constant_functor<HMM::Probability_t>(inf);
    const auto min = thrust::minimum<HMM::Probability_t>();
    const auto plus = thrust::plus<HMM::Probability_t>();

    cusp::multiply(lhs, rhs, res, init, plus, min);
}

HMM::Prob_vec_t Dev_mat_to_Prob_vec(const Dev_mat& mat) {
#ifndef NDEBUG
    for (auto c : mat.column_indices) {
        if (c != 0) {
            std::cerr << "Internal implementation error! CUSP COO matrix is not a column!\n";
        }
    }
#endif
    auto res = HMM::Prob_vec_t(mat.num_rows, HMM::Probability_t());
    for (size_t i = 0; i < mat.num_entries; ++i) {
        res[mat.row_indices[i]] = mat.values[i];
    }
    return res;
}

bool is_not_zero_prob(HMM::Probability_t x) {
    // Transformed zero probability is infinity, i.e. std::log2(0)
    return !HMM::almost_equal(x, std::numeric_limits<HMM::Probability_t>::infinity());
}
} // namespace CUSP_helper
