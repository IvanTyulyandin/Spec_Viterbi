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

bool compare_two_answers(const HMM::Mod_prob_vec_t& lhs, const HMM::Mod_prob_vec_t& rhs) {
    auto are_equal = true;
    for (size_t i = 0; i < lhs.size(); ++i) {
        are_equal &= HMM::almost_equal(lhs[i], rhs[i]);
    }
    are_equal &= (lhs.size() == rhs.size());

    return are_equal;
}

bool test_impl(const Viterbi_impl& impl) {
    auto res = impl.run_Viterbi(hmm, seq);
    return compare_two_answers(res, expected_res);
}

bool test_spec_impl(Viterbi_spec_impl& impl) {
    impl.spec_with(hmm);
    auto res = impl.run_Viterbi_spec(seq);
    return compare_two_answers(res, expected_res);
}

} // namespace test_helper
