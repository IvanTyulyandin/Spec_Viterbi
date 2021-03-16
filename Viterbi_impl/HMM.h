#pragma once

#include <cmath>
#include <vector>

class HMM {
  public:
    using Probability_t = float;
    using Index_t = size_t;
    using Emit_t = size_t;
    using Prob_vec_t = std::vector<Probability_t>;
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
    Prob_vec_t trans_probs;
    Prob_vec_t emissions;
    Index_t non_zero_start_probs;
    Prob_vec_t start_probabilities;

    // Functions to work with Probability_t

    static bool almost_equal(HMM::Probability_t x, HMM::Probability_t y) {
        return std::fabs(x - y) <= 0.0001;
    }

    static HMM::Probability_t to_neg_log(HMM::Probability_t x) { return -1 * std::log2(x); }
};
