#include "CUSP_impl.h"

// Should be first header to define device
#include "CUSP_helper.h"

#include <cusp/print.h>

HMM::Prob_vec_t CUSP_impl::run_Viterbi(const HMM& hmm, const HMM::Emit_seq_t& seq) const {
    // Define column for start probabilities
    auto start_pr_host = CUSP_helper::Host_mat(hmm.states_num, 1, hmm.non_zero_start_probs);
    for (size_t i = 0, j = 0; i < hmm.start_probabilities.size(); ++i) {
        if (CUSP_helper::is_not_zero_prob(hmm.start_probabilities[i])) {
            start_pr_host.row_indices[j] = i;
            start_pr_host.column_indices[j] = 0;
            start_pr_host.values[j] = hmm.start_probabilities[i];
            ++j;
        }
    }
    start_pr_host.sort_by_row_and_column();

    // Transposed transition matrix
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
        auto offset = i;
        for (size_t j = 0; j < hmm.states_num; ++j) {
            m.row_indices[j] = j;
            m.column_indices[j] = j;
            m.values[j] = hmm.emissions[offset];
            offset += hmm.emit_num;
        }
        m.sort_by_row_and_column();
    }

    // Move data to the device side
    auto transp_tr_dev = CUSP_helper::Dev_mat(transp_tr_host);
    auto start_pr_dev = CUSP_helper::Dev_mat(start_pr_host);
    auto emit_mat_vec_dev =
        std::vector<CUSP_helper::Dev_mat>(emit_mat_vec_host.begin(), emit_mat_vec_host.end());

    // Start Viterbi
    auto res = CUSP_helper::Dev_mat();
    CUSP_helper::min_plus_Dev_mat_multiply(emit_mat_vec_dev[seq[0]], start_pr_dev, res);

    for (size_t i = 1; i < seq.size(); ++i) {
        auto tmp = res;
        CUSP_helper::min_plus_Dev_mat_multiply(transp_tr_dev, res, tmp);
        CUSP_helper::min_plus_Dev_mat_multiply(emit_mat_vec_dev[seq[i]], tmp, res);
    }

    return CUSP_helper::Dev_mat_to_Prob_vec(res);
}
