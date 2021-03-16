#include "GraphBLAS_spec_impl.h"

#include "HMM.h"

using Obs_handler_t = std::unordered_map<std::vector<size_t>, GrB_Matrix, HMM::Emit_seq_hasher>;

namespace {

void free_obs_handler(Obs_handler_t& handler) {
    for (auto& [k, v] : handler) {
        GrB_Matrix_free(&v);
    }
}

void add_level(Obs_handler_t& prev, const std::vector<GrB_Matrix>& updater, HMM::Index_t mat_size,
               HMM::Index_t emit_num) {
    auto res = Obs_handler_t(prev.size() * emit_num);

    for (const auto& [k, v] : prev) {
        for (size_t i = 0; i < emit_num; ++i) {
            auto new_key = HMM::Emit_seq_t(k);
            new_key.push_back(i);

            auto handler = GrB_Matrix();
            auto info = GrB_Matrix_new(&handler, GrB_FP32, mat_size, mat_size);
            GraphBLAS_helper::check_for_error(info);

            info = GrB_mxm(handler, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP32, updater[i], v,
                           GrB_NULL);
            GraphBLAS_helper::check_for_error(info);

            res.insert({std::move(new_key), handler});
        }
    }

    free_obs_handler(prev);
    prev = std::move(res);
}

} // namespace

GraphBLAS_spec_impl::GraphBLAS_spec_impl(const HMM& hmm, size_t level)
    : precalc_obs_handlers(), level(level) {
    initializer(hmm, level);
}

void GraphBLAS_spec_impl::spec_with(const HMM& hmm) {
    deleter();
    initializer(hmm, level);
}

HMM::Prob_vec_t GraphBLAS_spec_impl::run_Viterbi_spec(const HMM::Emit_seq_t& seq) const {
    auto result = GrB_Matrix();

    // Start Viterbi algorithm for seq[0]
    auto info = GrB_Matrix_dup(&result, emit_pr_x_start_pr[seq[0]]);
    GraphBLAS_helper::check_for_error(info);

    auto next_probs = GrB_Matrix();
    info = GrB_Matrix_new(&next_probs, GrB_FP32, states_num, 1);
    GraphBLAS_helper::check_for_error(info);

    // Viterbi algorithm main part

    // Use precalculated matrices while it is possible
    auto obs_handler_matrix = GrB_Matrix();

    auto i = size_t(1);
    if (level > 1) {
        while ((seq.size() - i) >= level) {
            auto obs_to_handle = HMM::Emit_seq_t(level, 0);
            for (size_t j = 0; j < level; ++j, ++i) {
                obs_to_handle[j] = seq[i];
            }
            // A result must exist in precalc_obs_handlers by construction
            obs_handler_matrix = precalc_obs_handlers.at(obs_to_handle);

            info = GrB_mxm(next_probs, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP32,
                           obs_handler_matrix, result, GrB_NULL);
            GraphBLAS_helper::check_for_error(info);
            GrB_Matrix_wait(&next_probs);
            std::swap(next_probs, result);
        }
    }

    // Handle the seq tail
    for (; i < seq.size(); ++i) {
        info = GrB_mxm(next_probs, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP32,
                       emit_pr_x_trans_pr[seq[i]], result, GrB_NULL);
        GraphBLAS_helper::check_for_error(info);
        GrB_Matrix_wait(&next_probs);
        std::swap(next_probs, result);
    }

    auto res = GraphBLAS_helper::GrB_Matrix_to_Prob_vec(result);

    GrB_Matrix_free(&next_probs);
    GrB_Matrix_free(&result);

    return res;
}

GraphBLAS_spec_impl::~GraphBLAS_spec_impl() { deleter(); }

// Private methods implementation

void GraphBLAS_spec_impl::initializer(const HMM& hmm, size_t level) {
    states_num = hmm.states_num;

    // Transposed HMM transition matrix
    auto transposed_transitions = GrB_Matrix();
    auto info = GrB_Matrix_new(&transposed_transitions, GrB_FP32, states_num, states_num);
    GraphBLAS_helper::check_for_error(info);

    info =
        GrB_Matrix_build_FP32(transposed_transitions, hmm.trans_cols.data(), hmm.trans_rows.data(),
                              hmm.trans_probs.data(), hmm.trans_num, GrB_FIRST_FP32);
    GraphBLAS_helper::check_for_error(info);

    // Read info about states with
    // non zero probabilities to be start

    auto n_zeroes_ind = std::vector<GrB_Index>(states_num, 0);
    auto from_0_to_n_ind = std::vector<GrB_Index>(states_num);
    for (size_t i = 0; i < states_num; ++i) {
        from_0_to_n_ind[i] = i;
    }

    auto start_probs = GrB_Matrix();
    info = GrB_Matrix_new(&start_probs, GrB_FP32, states_num, 1);
    GraphBLAS_helper::check_for_error(info);
    info = GrB_Matrix_build_FP32(start_probs, from_0_to_n_ind.data(), n_zeroes_ind.data(),
                                 hmm.start_probabilities.data(), hmm.states_num, GrB_FIRST_FP32);
    GraphBLAS_helper::check_for_error(info);

    emit_pr_x_start_pr = std::vector<GrB_Matrix>(hmm.emit_num);
    emit_pr_x_trans_pr = std::vector<GrB_Matrix>(hmm.emit_num);
    for (size_t i = 0; i < hmm.emit_num; ++i) {
        info = GrB_Matrix_new(&(emit_pr_x_start_pr[i]), GrB_FP32, states_num, 1);
        GraphBLAS_helper::check_for_error(info);
        info = GrB_Matrix_new(&(emit_pr_x_trans_pr[i]), GrB_FP32, states_num, states_num);
        GraphBLAS_helper::check_for_error(info);
    }

    auto emit_data = HMM::Prob_vec_t(states_num);
    auto emit_probs_diag_mat = GrB_Matrix();
    info = GrB_Matrix_new(&emit_probs_diag_mat, GrB_FP32, states_num, states_num);
    GraphBLAS_helper::check_for_error(info);

    for (size_t i = 0; i < hmm.emit_num; ++i) {

        auto offset = i;
        for (size_t j = 0; j < states_num; ++j) {
            emit_data[j] = hmm.emissions[offset];
            offset += hmm.emit_num;
        }

        info = GrB_Matrix_build_FP32(emit_probs_diag_mat, from_0_to_n_ind.data(),
                                     from_0_to_n_ind.data(), emit_data.data(), hmm.states_num,
                                     GrB_FIRST_FP32);
        GraphBLAS_helper::check_for_error(info);

        info = GrB_mxm(emit_pr_x_start_pr[i], GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP32,
                       emit_probs_diag_mat, start_probs, GrB_NULL);
        GraphBLAS_helper::check_for_error(info);

        info = GrB_mxm(emit_pr_x_trans_pr[i], GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP32,
                       emit_probs_diag_mat, transposed_transitions, GrB_NULL);
        GraphBLAS_helper::check_for_error(info);

        info = GrB_Matrix_clear(emit_probs_diag_mat);
        GraphBLAS_helper::check_for_error(info);
    }

    GrB_Matrix_free(&transposed_transitions);
    GrB_Matrix_free(&start_probs);
    GrB_Matrix_free(&emit_probs_diag_mat);

    // Set up handlers
    if (level > 1) {
        for (size_t i = 0; i < hmm.emit_num; ++i) {
            auto copy = GrB_Matrix();
            info = GrB_Matrix_dup(&copy, emit_pr_x_trans_pr[i]);
            GraphBLAS_helper::check_for_error(info);

            precalc_obs_handlers.insert({HMM::Emit_seq_t(1, i), copy});
        }

        // It is already at level 1
        for (size_t i = 1; i < level; ++i) {
            add_level(precalc_obs_handlers, emit_pr_x_trans_pr, states_num, hmm.emit_num);
        }
    }
}

void GraphBLAS_spec_impl::deleter() {
    // Free internal matrices
    free_obs_handler(precalc_obs_handlers);
    precalc_obs_handlers = Obs_handler_t();

    for (auto& m : emit_pr_x_start_pr) {
        GrB_Matrix_free(&m);
    }
    emit_pr_x_start_pr = std::vector<GrB_Matrix>();

    for (auto& m : emit_pr_x_trans_pr) {
        GrB_Matrix_free(&m);
    }
    emit_pr_x_trans_pr = std::vector<GrB_Matrix>();
}
