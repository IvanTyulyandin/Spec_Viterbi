#pragma once

#include "HMM.h"

// Viterbi specialized algorithm implementation class
class Viterbi_spec_impl {
  public:
    virtual void spec_with(const HMM& hmm) = 0;

    [[nodiscard]] virtual HMM::Mod_prob_vec_t
    run_Viterbi_spec(const HMM::Emit_seq_t& seq) const = 0;

    virtual ~Viterbi_spec_impl() = default;

    size_t get_level() const { return level; }

  protected:
    // "level" is a number of observation handlers to combine and precalculate
    // checkout any of the specialized versions files, i.e. *_spec_impl.cpp, for more details
    size_t level;
};
