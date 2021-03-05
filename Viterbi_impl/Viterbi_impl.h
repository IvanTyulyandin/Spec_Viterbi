#pragma once

#include "HMM.h"

// Viterbi algorithm implementation class
class Viterbi_impl {
  public:
    [[nodiscard]] virtual HMM::Prob_vec_t run_Viterbi(const HMM& hmm,
                                                      const HMM::Emit_seq_t& seq) const = 0;
};
