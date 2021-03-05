#include "../Viterbi_impl/GraphBLAS_impl.h"
#include "../Viterbi_impl/GraphBLAS_manager.h"
#include "../Viterbi_impl/data_reader.h"

#include <iostream>

int main() {

    launch_GraphBLAS();

    auto hmm = read_HMM("../chmm_files/test_chmm.chmm");
    auto seq = read_emit_seq("../ess_files/test_seq.ess")[0];

    auto Viterbi_impl = GraphBLAS_impl();
    auto res = Viterbi_impl.run_Viterbi(hmm, seq);
    auto expected_res = HMM::Prob_vec_t{25.6574, 24.4874};

    auto is_test_passed = true;
    for (size_t i = 0; i < expected_res.size(); ++i) {
        is_test_passed &= HMM::almost_equal(res[i], expected_res[i]);
    }

    stop_GraphBLAS();

    return 1 - static_cast<int>(is_test_passed);
}
