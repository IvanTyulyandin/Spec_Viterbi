#pragma once

#include "GraphBLAS_manager.h"
#include "Viterbi_spec_impl.h"

#include <unordered_map>

class GraphBLAS_spec_impl : public Viterbi_spec_impl {
  public:
    using Obs_handler_t = std::unordered_map<std::vector<size_t>, GrB_Matrix, HMM::Emit_seq_hasher>;

    // "level" is the maximum depth for precalc_observation_handlers
    // if level == 1, it makes no sence, same matrices are stored in emit_pr_x_trans_pr
    // if level == 2, it will save all possible (emit_x_tr * emit_x_tr)
    // and so on
    explicit GraphBLAS_spec_impl(const HMM& hmm, size_t level);

    [[nodiscard]] HMM::Prob_vec_t run_Viterbi_spec(const HMM::Emit_seq_t& seq) const override;

    ~GraphBLAS_spec_impl();

  private:
    std::vector<GrB_Matrix> emit_pr_x_start_pr;
    std::vector<GrB_Matrix> emit_pr_x_trans_pr;
    Obs_handler_t precalc_obs_handlers;
    HMM::Index_t states_num;
    size_t level;
};
