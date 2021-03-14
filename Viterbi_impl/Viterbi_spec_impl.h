#pragma once

#include "HMM.h"

// Viterbi specialized algorithm implementation class
class Viterbi_spec_impl {
  public:
    virtual void spec_with(const HMM& hmm) = 0;

    [[nodiscard]] virtual HMM::Prob_vec_t run_Viterbi_spec(const HMM::Emit_seq_t& seq) const = 0;

    virtual ~Viterbi_spec_impl() = default;
};
