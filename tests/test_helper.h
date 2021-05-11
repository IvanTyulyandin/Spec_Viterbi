#pragma once

#include "../Viterbi_impl/Viterbi_impl.h"
#include "../Viterbi_impl/Viterbi_spec_impl.h"
#include "../Viterbi_impl/data_reader.h"

#include <iostream>

namespace test_helper {

const auto test_chmms_path = std::string("../chmm_files/test_chmms/");
const auto chmm_postfix = std::string("_test_chmm.chmm");

const auto test_ess_path = std::string("../ess_files/test_sequences/");
const auto ess_postfix = std::string("_test_seq.ess");

const auto expected_results =
    std::vector({HMM::Mod_prob_vec_t{25.6574, 24.4874, HMM::to_modified_prob(0)}});

constexpr auto LEVELS_TO_TEST = 7;

bool compare_two_answers(const HMM::Mod_prob_vec_t& lhs, const HMM::Mod_prob_vec_t& rhs) {
    auto are_equal = true;
    for (size_t i = 0; i < lhs.size(); ++i) {
        are_equal &= HMM::almost_equal(lhs[i], rhs[i]);
        if (!are_equal) {
            // std::cerr << lhs[i] << ' ' << rhs[i] << '\n';
            return false;
        }
    }
    // Uncomment if wish to print results
    // for (size_t i = 0; i < lhs.size(); ++i) {
    //     std::cout << lhs[i] << ' ' << rhs[i] << '\n';
    // }
    are_equal &= (lhs.size() == rhs.size());

    return are_equal;
}

bool test_impl(const Viterbi_impl& impl) {
    auto is_passed = true;
    for (size_t i = 0; i < expected_results.size(); ++i) {
        auto chmm = read_HMM(test_chmms_path + std::to_string(i) + chmm_postfix);
        auto seq = read_emit_seq(test_ess_path + std::to_string(i) + ess_postfix)[0];
        auto res = impl.run_Viterbi(chmm, seq);
        is_passed &= compare_two_answers(res, expected_results[i]);
        if (!is_passed) {
            std::cerr << "test_impl fail " << i << '\n';
            return false;
        }
    }
    return is_passed;
}

bool test_spec_impl(Viterbi_spec_impl& impl) {
    auto is_passed = true;
    for (size_t i = 0; i < expected_results.size(); ++i) {
        auto chmm = read_HMM(test_chmms_path + std::to_string(i) + chmm_postfix);
        auto seq = read_emit_seq(test_ess_path + std::to_string(i) + ess_postfix)[0];
        impl.spec_with(chmm);
        auto res = impl.run_Viterbi_spec(seq);
        is_passed &= compare_two_answers(res, expected_results[i]);
        if (!is_passed) {
            std::cerr << "test_spec_impl fail " << i << '\n';
            return false;
        }
    }
    return is_passed;
}

} // namespace test_helper
