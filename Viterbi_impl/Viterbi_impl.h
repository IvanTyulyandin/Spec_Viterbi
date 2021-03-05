#pragma once

#include "HMM.h"

// Viterbi algorithm implementation class
class Viterbi_impl {
  public:
    virtual HMM::Probability_t run_Viterbi(const HMM& hmm,
                                           const HMM::Seq_vec_t& emit_vec) const = 0;
};
