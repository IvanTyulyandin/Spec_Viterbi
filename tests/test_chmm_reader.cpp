#include "../Viterbi_impl/data_reader.h"

int main() {
    auto hmm = read_HMM("../chmm_files/test_chmm.chmm");
    auto is_test_passed =
        (hmm.states_num == 2 && hmm.non_zero_start_probs == 2 &&
         HMM::almost_equal(hmm.start_probabilities[0], HMM::to_neg_log(0.5)) &&
         HMM::almost_equal(hmm.start_probabilities[1], HMM::to_neg_log(0.5)) && hmm.emit_num == 4 &&
         HMM::almost_equal(hmm.emissions[0], HMM::to_neg_log(0.2)) &&
         HMM::almost_equal(hmm.emissions[1], HMM::to_neg_log(0.3)) &&
         HMM::almost_equal(hmm.emissions[2], HMM::to_neg_log(0.3)) &&
         HMM::almost_equal(hmm.emissions[3], HMM::to_neg_log(0.2)) &&
         HMM::almost_equal(hmm.emissions[4], HMM::to_neg_log(0.3)) &&
         HMM::almost_equal(hmm.emissions[5], HMM::to_neg_log(0.2)) &&
         HMM::almost_equal(hmm.emissions[6], HMM::to_neg_log(0.2)) &&
         HMM::almost_equal(hmm.emissions[7], HMM::to_neg_log(0.3)) && hmm.trans_num == 4 &&
         hmm.trans_rows == HMM::Index_vec_t{0, 0, 1, 1} &&
         hmm.trans_cols == HMM::Index_vec_t{0, 1, 0, 1} &&
         HMM::almost_equal(hmm.trans_probs[0], HMM::to_neg_log(0.5)) &&
         HMM::almost_equal(hmm.trans_probs[1], HMM::to_neg_log(0.5)) &&
         HMM::almost_equal(hmm.trans_probs[2], HMM::to_neg_log(0.4)) &&
         HMM::almost_equal(hmm.trans_probs[3], HMM::to_neg_log(0.6)));

    return 1 - static_cast<int>(is_test_passed);
}
