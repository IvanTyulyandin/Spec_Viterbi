#pragma once

#include <cmath>
#include <vector>

class HMM {
  public:
    using Probability_t = float;
    using Mod_prob_t = float;
    using Index_t = size_t;
    using Emit_t = size_t;
    using Mod_prob_vec_t = std::vector<Mod_prob_t>;
    using Index_vec_t = std::vector<Index_t>;
    using Emit_seq_t = std::vector<Emit_t>;
    using Emit_seq_vec_t = std::vector<Emit_seq_t>;

    struct Emit_seq_hasher {
        std::size_t operator()(const HMM::Emit_seq_t& vec) const {
            auto seed = vec.size();
            for (auto& i : vec) {
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    Index_t states_num;
    Index_t emit_num;
    Index_t trans_num;
    // encode states and emit symbols as numbers
    Index_vec_t trans_rows;
    Index_vec_t trans_cols;
    Mod_prob_vec_t trans_probs;
    Mod_prob_vec_t emissions;
    Index_t non_zero_start_probs;
    Mod_prob_vec_t start_probabilities;

    // Functions to work with Probability_t
    static constexpr auto zero_prob = std::numeric_limits<HMM::Probability_t>::infinity();

    static bool almost_equal(HMM::Probability_t x, HMM::Probability_t y) {
        const auto is_both_inf = (zero_prob == x) && (zero_prob == y);
        return is_both_inf || std::fabs(x - y) <= 0.0001;
    }

    static HMM::Mod_prob_t to_modified_prob(HMM::Probability_t x) {
        if (x != zero_prob) {
            return -1 * std::log2(x);
        } else {
            return zero_prob;
        }
    }
};
