#include "../Viterbi_impl/data_reader.h"

int main() {
    auto hmm = read_HMM("../chmm_files/test_chmm.chmm");
    auto is_test_passed =
        (hmm.states_num == 3 && hmm.non_zero_start_probs == 2 &&
         HMM::almost_equal(hmm.start_probabilities[0], HMM::to_modified_prob(0.5)) &&
         HMM::almost_equal(hmm.start_probabilities[1], HMM::to_modified_prob(0.5)) &&
         HMM::almost_equal(hmm.start_probabilities[2], HMM::to_modified_prob(0)) &&
         hmm.emit_num == 4 && hmm.emissions.size() == 4 &&
         HMM::almost_equal(hmm.emissions[0][0], HMM::to_modified_prob(0.2)) &&
         HMM::almost_equal(hmm.emissions[1][0], HMM::to_modified_prob(0.3)) &&
         HMM::almost_equal(hmm.emissions[2][0], HMM::to_modified_prob(0.3)) &&
         HMM::almost_equal(hmm.emissions[3][0], HMM::to_modified_prob(0.2)) &&
         HMM::almost_equal(hmm.emissions[0][1], HMM::to_modified_prob(0.3)) &&
         HMM::almost_equal(hmm.emissions[1][1], HMM::to_modified_prob(0.2)) &&
         HMM::almost_equal(hmm.emissions[2][1], HMM::to_modified_prob(0.2)) &&
         HMM::almost_equal(hmm.emissions[3][1], HMM::to_modified_prob(0.3)) &&
         HMM::almost_equal(hmm.emissions[0][2], HMM::to_modified_prob(0.3)) &&
         HMM::almost_equal(hmm.emissions[1][2], HMM::to_modified_prob(0.2)) &&
         HMM::almost_equal(hmm.emissions[2][2], HMM::to_modified_prob(0.2)) &&
         HMM::almost_equal(hmm.emissions[3][2], HMM::to_modified_prob(0.3)) && hmm.trans_num == 4 &&
         hmm.trans_rows == HMM::Index_vec_t{0, 0, 1, 1} &&
         hmm.trans_cols == HMM::Index_vec_t{0, 1, 0, 1} &&
         HMM::almost_equal(hmm.trans_probs[0], HMM::to_modified_prob(0.5)) &&
         HMM::almost_equal(hmm.trans_probs[1], HMM::to_modified_prob(0.5)) &&
         HMM::almost_equal(hmm.trans_probs[2], HMM::to_modified_prob(0.4)) &&
         HMM::almost_equal(hmm.trans_probs[3], HMM::to_modified_prob(0.6)));

    return 1 - static_cast<int>(is_test_passed);
}
