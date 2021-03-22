#include "CUSP_spec_impl.h"

#include "HMM.h"

using Obs_handler_t = CUSP_spec_impl::Obs_handler_t;

namespace {

void add_level(Obs_handler_t& prev, const std::vector<CUSP_helper::Dev_mat>& updater,
               HMM::Index_t emit_num) {
    auto res = Obs_handler_t(prev.size() * emit_num);

    for (const auto& [k, v] : prev) {
        for (size_t i = 0; i < emit_num; ++i) {
            auto new_key = HMM::Emit_seq_t(k);
            new_key.push_back(i);

            auto handler = CUSP_helper::Dev_mat();
            CUSP_helper::min_plus_Dev_mat_multiply(updater[i], v, handler);

            res.insert({std::move(new_key), handler});
        }
    }

    prev = std::move(res);
}

} // namespace

CUSP_spec_impl::CUSP_spec_impl(const HMM& hmm, size_t level)
    : precalc_obs_handlers(), level(level) {
    initializer(hmm, level);
}

void CUSP_spec_impl::spec_with(const HMM& hmm) {
    deleter();
    initializer(hmm, level);
}

HMM::Mod_prob_vec_t CUSP_spec_impl::run_Viterbi_spec(const HMM::Emit_seq_t& seq) const {
    // Start Viterbi algorithm for seq[0]
    auto result = emit_pr_x_start_pr[seq[0]];

    auto next_probs = CUSP_helper::Dev_mat();

    // Viterbi algorithm main part

    // Use precalculated matrices while it is possible
    auto obs_handler_matrix = CUSP_helper::Dev_mat();

    auto i = size_t(1);
    if (level > 1) {
        while ((seq.size() - i) >= level) {
            auto obs_to_handle = HMM::Emit_seq_t(level, 0);
            for (size_t j = 0; j < level; ++j, ++i) {
                obs_to_handle[j] = seq[i];
            }
            // A result must exist in precalc_obs_handlers by construction
            obs_handler_matrix = precalc_obs_handlers.at(obs_to_handle);

            CUSP_helper::min_plus_Dev_mat_multiply(obs_handler_matrix, result, next_probs);

            std::swap(next_probs, result);
        }
    }

    // Handle the seq tail
    for (; i < seq.size(); ++i) {
        CUSP_helper::min_plus_Dev_mat_multiply(emit_pr_x_trans_pr[seq[i]], result, next_probs);

        std::swap(next_probs, result);
    }

    return CUSP_helper::Dev_mat_to_Prob_vec(result);
}

CUSP_spec_impl::~CUSP_spec_impl() { deleter(); }

// Private methods implementation

void CUSP_spec_impl::initializer(const HMM& hmm, size_t level) {
    states_num = hmm.states_num;

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

    // Set up emit_pr_x_start_pr and emit_pr_x_trans_pr
    emit_pr_x_start_pr = std::vector<CUSP_helper::Dev_mat>(hmm.emit_num);
    emit_pr_x_trans_pr = std::vector<CUSP_helper::Dev_mat>(hmm.emit_num);
    for (size_t i = 0; i < hmm.emit_num; ++i) {
        CUSP_helper::min_plus_Dev_mat_multiply(emit_mat_vec_dev[i], start_pr_dev,
                                               emit_pr_x_start_pr[i]);
        CUSP_helper::min_plus_Dev_mat_multiply(emit_mat_vec_dev[i], transp_tr_dev,
                                               emit_pr_x_trans_pr[i]);
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

void CUSP_spec_impl::deleter() {
    // Clear internal matrices
    precalc_obs_handlers = Obs_handler_t();
    emit_pr_x_start_pr = std::vector<CUSP_helper::Dev_mat>();
    emit_pr_x_trans_pr = std::vector<CUSP_helper::Dev_mat>();
}
