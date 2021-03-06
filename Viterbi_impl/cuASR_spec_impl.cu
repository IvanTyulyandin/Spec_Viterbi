#include "cuASR_spec_impl.h"

#include "HMM.h"

using Obs_handler_t = cuASR_spec_impl::Obs_handler_t;

namespace {

void add_level(Obs_handler_t& prev, const std::vector<cuASR_helper::Dev_mat>& updater,
               HMM::Index_t emit_num) {
    auto res = Obs_handler_t(prev.size() * emit_num);

    for (const auto& kv : prev) {
        for (size_t i = 0; i < emit_num; ++i) {
            auto new_key = HMM::Emit_seq_t(kv.first);
            new_key.push_back(i);

            auto handler = cuASR_helper::Dev_mat();
            cuASR_helper::min_plus_Dev_mat_multiply(updater[i], kv.second, handler);

            res.insert({std::move(new_key), handler});
        }
    }

    prev = std::move(res);
}

} // namespace

cuASR_spec_impl::cuASR_spec_impl(const HMM& hmm, size_t level)
    : Viterbi_spec_impl(level), precalc_obs_handlers() {
    initializer(hmm, level);
}

void cuASR_spec_impl::spec_with(const HMM& hmm) {
    deleter();
    initializer(hmm, level);
}

HMM::Mod_prob_vec_t cuASR_spec_impl::run_Viterbi_spec(const HMM::Emit_seq_t& seq) const {
    // Start Viterbi algorithm for seq[0]
    auto result = emit_pr_x_start_pr[seq[0]];

    auto next_probs = cuASR_helper::Dev_mat();

    // Viterbi algorithm main part

    // Use precalculated matrices while it is possible
    auto obs_handler_matrix = cuASR_helper::Dev_mat();

    auto i = size_t(1);
    if (level > 1) {
        while ((seq.size() - i) >= level) {
            auto obs_to_handle = HMM::Emit_seq_t(level, 0);
            for (size_t j = 0; j < level; ++j, ++i) {
                obs_to_handle[j] = seq[i];
            }
            // A result must exist in precalc_obs_handlers by construction
            obs_handler_matrix = precalc_obs_handlers.at(obs_to_handle);

            cuASR_helper::min_plus_Dev_mat_multiply(obs_handler_matrix, result, next_probs);

            std::swap(next_probs, result);
        }
    }

    // Handle the seq tail
    for (; i < seq.size(); ++i) {
        cuASR_helper::min_plus_Dev_mat_multiply(emit_pr_x_trans_pr[seq[i]], result, next_probs);

        std::swap(next_probs, result);
    }

    return cuASR_helper::Dev_mat_to_Prob_vec(result);
}

cuASR_spec_impl::~cuASR_spec_impl() { deleter(); }

// Private methods implementation

void cuASR_spec_impl::initializer(const HMM& hmm, size_t level) {

    auto start_pr_dev = cuASR_helper::Dev_mat();
    auto transp_tr_dev = cuASR_helper::Dev_mat();
    auto emit_mat_vec_dev = std::vector<cuASR_helper::Dev_mat>();
    cuASR_helper::init_matrices_from_HMM(hmm, start_pr_dev, transp_tr_dev, emit_mat_vec_dev);
    cuASR_helper::validate_Dev_mat_ptr(start_pr_dev, std::string("start_pr_dev"));
    cuASR_helper::validate_Dev_mat_ptr(transp_tr_dev, std::string("transp_tr_dev"));

    // Set up emit_pr_x_start_pr and emit_pr_x_trans_pr
    emit_pr_x_start_pr = std::vector<cuASR_helper::Dev_mat>(hmm.emit_num);
    emit_pr_x_trans_pr = std::vector<cuASR_helper::Dev_mat>(hmm.emit_num);
    for (size_t i = 0; i < hmm.emit_num; ++i) {
        cuASR_helper::validate_Dev_mat_ptr(emit_mat_vec_dev[i], std::string("emit_mat_vec_dev[") +
                                                                    std::to_string(i) +
                                                                    std::string("]"));
        cuASR_helper::min_plus_Dev_mat_multiply(emit_mat_vec_dev[i], start_pr_dev,
                                                emit_pr_x_start_pr[i]);
        cuASR_helper::validate_Dev_mat_ptr(emit_pr_x_start_pr[i],
                                           std::string("emit_pr_x_start_pr[") + std::to_string(i) +
                                               std::string("]"));
        cuASR_helper::min_plus_Dev_mat_multiply(emit_mat_vec_dev[i], transp_tr_dev,
                                                emit_pr_x_trans_pr[i]);
        cuASR_helper::validate_Dev_mat_ptr(emit_pr_x_trans_pr[i],
                                           std::string("emit_pr_x_trans_pr[") + std::to_string(i) +
                                               std::string("]"));
    }

    // Set up handlers
    if (level > 1) {
        for (size_t i = 0; i < hmm.emit_num; ++i) {
            precalc_obs_handlers.insert({HMM::Emit_seq_t(1, i), emit_pr_x_trans_pr[i]});
        }

        // It is already at level 1
        for (size_t i = 1; i < level; ++i) {
            add_level(precalc_obs_handlers, emit_pr_x_trans_pr, hmm.emit_num);
        }
    }
}

void cuASR_spec_impl::deleter() {
    // Clear internal matrices
    precalc_obs_handlers = Obs_handler_t();
    emit_pr_x_start_pr = std::vector<cuASR_helper::Dev_mat>();
    emit_pr_x_trans_pr = std::vector<cuASR_helper::Dev_mat>();
}
