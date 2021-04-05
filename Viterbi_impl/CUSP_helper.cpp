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

HMM::Mod_prob_vec_t Dev_mat_to_Prob_vec(const Dev_mat& mat) {
#ifndef NDEBUG
    for (auto c : mat.column_indices) {
        if (c != 0) {
            std::cerr << "Internal implementation error! CUSP COO matrix is not a column!\n";
        }
    }
#endif
    auto res = HMM::Mod_prob_vec_t(mat.num_rows, HMM::to_modified_prob(0));
    for (size_t i = 0; i < mat.num_entries; ++i) {
        res[mat.row_indices[i]] = mat.values[i];
    }
    return res;
}

void init_matrices_from_HMM(const HMM& hmm, Dev_mat& start_pr_dev, Dev_mat& transp_tr_dev,
                            std::vector<Dev_mat>& emit_mat_vec_dev) {
    // Define column for start probabilities
    auto start_pr_host = Host_mat(hmm.states_num, 1, hmm.non_zero_start_probs);
    for (size_t i = 0, j = 0; i < hmm.start_probabilities.size(); ++i) {
        if (HMM::is_not_zero_mod_prob(hmm.start_probabilities[i])) {
            start_pr_host.row_indices[j] = i;
            start_pr_host.column_indices[j] = 0;
            start_pr_host.values[j] = hmm.start_probabilities[i];
            ++j;
        }
    }
    start_pr_host.sort_by_row_and_column();

    // Transposed HMM transition matrix
    auto transp_tr_host = CUSP_helper::Host_mat(hmm.states_num, hmm.states_num, hmm.trans_num);
    for (size_t i = 0; i < hmm.trans_num; ++i) {
        transp_tr_host.row_indices[i] = hmm.trans_cols[i];
        transp_tr_host.column_indices[i] = hmm.trans_rows[i];
        transp_tr_host.values[i] = hmm.trans_probs[i];
    }
    transp_tr_host.sort_by_row_and_column();

    // Emit diagonal matrices
    auto emit_mat_vec_host = std::vector<CUSP_helper::Host_mat>(hmm.emit_num);
    for (size_t i = 0; i < hmm.emit_num; ++i) {
        auto& m = emit_mat_vec_host[i];
        // May have some zeroes instead of non zeroes
        m = CUSP_helper::Host_mat(hmm.states_num, hmm.states_num, hmm.states_num);
        for (size_t j = 0; j < hmm.states_num; ++j) {
            m.row_indices[j] = j;
            m.column_indices[j] = j;
            m.values[j] = hmm.emissions[i][j];
        }
        m.sort_by_row_and_column();
    }

    // Move data to the device side
    transp_tr_dev = Dev_mat(transp_tr_host);
    start_pr_dev = Dev_mat(start_pr_host);
    emit_mat_vec_dev = std::vector<Dev_mat>(emit_mat_vec_host.begin(), emit_mat_vec_host.end());

    return;
}

} // namespace CUSP_helper
