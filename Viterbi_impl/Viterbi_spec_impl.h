#pragma once

#include "HMM.h"

// Viterbi specialized algorithm implementation class
class Viterbi_spec_impl {
  public:
    virtual void spec_with(const HMM& hmm) = 0;

    virtual HMM::Probability_t run_Viterbi_spec(const HMM::Seq_vec_t& emit_vec) const = 0;
};
