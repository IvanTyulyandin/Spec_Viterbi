#pragma once

#include "Viterbi_impl.h"

class CUSP_impl : public Viterbi_impl {
  public:
    [[nodiscard]] HMM::Mod_prob_vec_t run_Viterbi(const HMM& hmm,
                                                  const HMM::Emit_seq_t& seq) const override;
};
