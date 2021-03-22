#include "../Viterbi_impl/GraphBLAS_helper.h"
#include "../Viterbi_impl/GraphBLAS_spec_impl.h"
#include "../Viterbi_impl/data_reader.h"

#include <iostream>

int main() {

    GraphBLAS_helper::get_instance().launch_GraphBLAS();

    auto hmm = read_HMM("../chmm_files/test_chmm.chmm");
    auto seq = read_emit_seq("../ess_files/test_seq.ess")[0];

    auto spec_lvl_1 = GraphBLAS_spec_impl(hmm, 1);
    auto spec_lvl_2 = GraphBLAS_spec_impl(hmm, 2);
    auto spec_lvl_3 = GraphBLAS_spec_impl(hmm, 3);

    auto res_lvl_1 = spec_lvl_1.run_Viterbi_spec(seq);
    auto res_lvl_2 = spec_lvl_2.run_Viterbi_spec(seq);
    auto res_lvl_3 = spec_lvl_3.run_Viterbi_spec(seq);

    auto expected_res = HMM::Mod_prob_vec_t{25.6574, 24.4874};

    auto is_test_passed = true;
    for (size_t i = 0; i < expected_res.size(); ++i) {
        is_test_passed &= HMM::almost_equal(res_lvl_1[i], expected_res[i]);
        is_test_passed &= HMM::almost_equal(res_lvl_2[i], expected_res[i]);
        is_test_passed &= HMM::almost_equal(res_lvl_3[i], expected_res[i]);
    }

    is_test_passed &= (res_lvl_1.size() == expected_res.size()) &&
                      (res_lvl_2.size() == expected_res.size()) &&
                      (res_lvl_3.size() == expected_res.size());

    return 1 - static_cast<int>(is_test_passed);
}
