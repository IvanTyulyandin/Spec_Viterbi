#include "CUSP_impl.h"

// Should be first header to define device
#include "CUSP_helper.h"

#include <cusp/print.h>

HMM::Mod_prob_vec_t CUSP_impl::run_Viterbi(const HMM& hmm, const HMM::Emit_seq_t& seq) const {

    auto start_pr_dev = CUSP_helper::Dev_mat();
    auto transp_tr_dev = CUSP_helper::Dev_mat();
    auto emit_mat_vec_dev = std::vector<CUSP_helper::Dev_mat>();
    CUSP_helper::init_matrices_from_HMM(hmm, start_pr_dev, transp_tr_dev, emit_mat_vec_dev);

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
