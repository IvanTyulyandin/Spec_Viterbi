#include "cuASR_impl.h"

#include "cuASR_helper.h"
#include <iostream>

HMM::Mod_prob_vec_t cuASR_impl::run_Viterbi(const HMM& hmm, const HMM::Emit_seq_t& seq) const {

    auto start_pr_dev = cuASR_helper::Dev_mat();
    auto transp_tr_dev = cuASR_helper::Dev_mat();
    auto emit_mat_vec_dev = std::vector<cuASR_helper::Dev_mat>();
    cuASR_helper::init_matrices_from_HMM(hmm, start_pr_dev, transp_tr_dev, emit_mat_vec_dev);

    // Start Viterbi
    auto res = cuASR_helper::Dev_mat(hmm.states_num, 1);
    cuASR_helper::min_plus_Dev_mat_multiply(emit_mat_vec_dev[seq[0]], start_pr_dev, res);

    for (size_t i = 1; i < seq.size(); ++i) {
        auto tmp = cuASR_helper::Dev_mat(hmm.states_num, 1);
        cuASR_helper::min_plus_Dev_mat_multiply(transp_tr_dev, res, tmp);
        cuASR_helper::min_plus_Dev_mat_multiply(emit_mat_vec_dev[seq[i]], tmp, res);
    }

    return cuASR_helper::Dev_mat_to_Prob_vec(res);
}
