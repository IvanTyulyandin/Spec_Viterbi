#include "../Viterbi_impl/CUSP_helper.h"
#include "../Viterbi_impl/CUSP_impl.h"
#include "../Viterbi_impl/data_reader.h"

#include <iostream>

int main() {

    auto hmm = read_HMM("../chmm_files/test_chmm.chmm");
    auto seq = read_emit_seq("../ess_files/test_seq.ess")[0];

    auto Viterbi_impl = CUSP_impl();
    auto res = Viterbi_impl.run_Viterbi(hmm, seq);
    auto expected_res = HMM::Mod_prob_vec_t{25.6574, 24.4874};

    auto is_test_passed = true;
    for (size_t i = 0; i < expected_res.size(); ++i) {
        std::cout << res[i] << '\n';
        is_test_passed &= HMM::almost_equal(res[i], expected_res[i]);
    }

    return 1 - static_cast<int>(is_test_passed);
}
