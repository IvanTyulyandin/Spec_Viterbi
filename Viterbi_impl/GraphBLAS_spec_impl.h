#pragma once

#include "Viterbi_spec_impl.h"

#include "GraphBLAS_helper.h"
#include <unordered_map>

class GraphBLAS_spec_impl : public Viterbi_spec_impl {
  public:
    using Obs_handler_t = std::unordered_map<std::vector<size_t>, GrB_Matrix, HMM::Emit_seq_hasher>;

    explicit GraphBLAS_spec_impl(const HMM& hmm, size_t level);

    explicit GraphBLAS_spec_impl(size_t level) : Viterbi_spec_impl(level), precalc_obs_handlers(){};

    [[nodiscard]] HMM::Mod_prob_vec_t run_Viterbi_spec(const HMM::Emit_seq_t& seq) const override;

    void spec_with(const HMM& hmm) override;

    ~GraphBLAS_spec_impl() override;

  private:
    std::vector<GrB_Matrix> emit_pr_x_start_pr;
    std::vector<GrB_Matrix> emit_pr_x_trans_pr;
    Obs_handler_t precalc_obs_handlers;
    HMM::Index_t states_num;

    void initializer(const HMM& hmm, size_t level);
    void deleter();
};
