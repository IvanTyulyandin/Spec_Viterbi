#include "GraphBLAS_impl.h"
#include "GraphBLAS_helper.h"

HMM::Prob_vec_t GraphBLAS_impl::run_Viterbi(const HMM& hmm, const HMM::Emit_seq_t& seq) const {
    // Define GraphBLAS matrices
    auto result = GrB_Matrix();

    // Prepare result to store data about start/current probabilities
    auto info = GrB_Matrix_new(&result, GrB_FP32, hmm.states_num, 1);
    GraphBLAS_helper::check_for_error(info);

    auto n_zeroes_ind = std::vector<GrB_Index>(hmm.states_num, 0);
    auto from_0_to_n_ind = std::vector<GrB_Index>(hmm.states_num);
    for (size_t i = 0; i < hmm.states_num; ++i) {
        from_0_to_n_ind[i] = i;
    }

    info = GrB_Matrix_build_FP32(result, from_0_to_n_ind.data(), n_zeroes_ind.data(),
                                 hmm.start_probabilities.data(), hmm.states_num, GrB_FIRST_FP32);
    GraphBLAS_helper::check_for_error(info);

    // Emission probabilities matrices
    auto em_probs = std::vector<GrB_Matrix>(hmm.emit_num);
    auto emit_data = HMM::Prob_vec_t(hmm.states_num);

    for (size_t i = 0; i < hmm.emit_num; ++i) {
        auto& m = em_probs[i];
        m = GrB_Matrix();
        info = GrB_Matrix_new(&m, GrB_FP32, hmm.states_num, hmm.states_num);
        GraphBLAS_helper::check_for_error(info);

        auto offset = i;
        for (size_t j = 0; j < hmm.states_num; ++j) {
            emit_data[j] = hmm.emissions[offset];
            offset += hmm.emit_num;
        }

        info = GrB_Matrix_build_FP32(m, from_0_to_n_ind.data(), from_0_to_n_ind.data(),
                                     emit_data.data(), hmm.states_num, GrB_FIRST_FP32);
        GraphBLAS_helper::check_for_error(info);
    }

    // Transposed HMM transition matrix
    auto transposed_transitions = GrB_Matrix();
    info = GrB_Matrix_new(&transposed_transitions, GrB_FP32, hmm.states_num, hmm.states_num);
    GraphBLAS_helper::check_for_error(info);

    info =
        GrB_Matrix_build_FP32(transposed_transitions, hmm.trans_cols.data(), hmm.trans_rows.data(),
                              hmm.trans_probs.data(), hmm.trans_num, GrB_FIRST_FP32);
    GraphBLAS_helper::check_for_error(info);

    // Matrices to store intermediate results
    auto prob_x_trans = GrB_Matrix();
    info = GrB_Matrix_new(&prob_x_trans, GrB_FP32, hmm.states_num, hmm.states_num);
    GraphBLAS_helper::check_for_error(info);

    auto next_probabilites = GrB_Matrix();
    info = GrB_Matrix_new(&next_probabilites, GrB_FP32, hmm.states_num, 1);
    GraphBLAS_helper::check_for_error(info);

    // Viterbi algorithm

    // Count emissions for first symbol and start probabilities
    info = GrB_mxm(next_probabilites, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP32,
                   em_probs[seq[0]], result, GrB_NULL);
    GraphBLAS_helper::check_for_error(info);
    GrB_Matrix_wait(&next_probabilites);
    std::swap(result, next_probabilites);

    for (size_t i = 1; i < seq.size(); ++i) {
        info = GrB_mxm(prob_x_trans, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP32,
                       em_probs[seq[i]], transposed_transitions, GrB_NULL);
        GraphBLAS_helper::check_for_error(info);
        GrB_Matrix_wait(&prob_x_trans);

        info = GrB_mxm(next_probabilites, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP32,
                       prob_x_trans, result, GrB_NULL);
        GraphBLAS_helper::check_for_error(info);
        GrB_Matrix_wait(&next_probabilites);

        std::swap(result, next_probabilites);
    }

    // Print matrix
    // SuiteSPARSE-specific (since GxB, not GrB)
    //
    // info = GxB_Matrix_fprint(result, "current_probabilities", GxB_COMPLETE, stdout);
    // check_for_error(info);

    auto res = GraphBLAS_helper::GrB_Matrix_to_Prob_vec(result);

    // Free resources
    for (auto& m : em_probs) {
        GrB_Matrix_free(&m);
    }
    GrB_Matrix_free(&transposed_transitions);
    GrB_Matrix_free(&prob_x_trans);
    GrB_Matrix_free(&next_probabilites);
    GrB_Matrix_free(&result);

    return res;
}
