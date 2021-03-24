#pragma once

#include "../Viterbi_impl/Viterbi_impl.h"
#include "../Viterbi_impl/Viterbi_spec_impl.h"
#include "../Viterbi_impl/data_reader.h"

#include <iostream>

namespace test_helper {

const auto hmm = read_HMM("../chmm_files/test_chmm.chmm");
const auto seq = read_emit_seq("../ess_files/test_seq.ess")[0];
const auto expected_res = HMM::Mod_prob_vec_t{25.6574, 24.4874, HMM::to_modified_prob(0)};

constexpr auto LEVELS_TO_TEST = 7;

bool compare_res_with_expected(const HMM::Mod_prob_vec_t& res) {
    auto are_equal = true;
    for (size_t i = 0; i < expected_res.size(); ++i) {
        are_equal &= HMM::almost_equal(res[i], expected_res[i]);
    }
    are_equal &= (res.size() == expected_res.size());

    return are_equal;
}

bool test_impl(const Viterbi_impl& impl) {
    auto res = impl.run_Viterbi(hmm, seq);
    return compare_res_with_expected(res);
}

bool test_spec_impl(Viterbi_spec_impl& impl) {
    impl.spec_with(hmm);
    auto res = impl.run_Viterbi_spec(seq);
    return compare_res_with_expected(res);
}

} // namespace test_helper
