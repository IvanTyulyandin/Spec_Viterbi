#pragma once

#include "Viterbi_spec_impl.h"

#include "CUSP_helper.h"
#include <unordered_map>

class CUSP_spec_impl : public Viterbi_spec_impl {
  public:
    using Obs_handler_t =
        std::unordered_map<std::vector<size_t>, CUSP_helper::Dev_mat, HMM::Emit_seq_hasher>;

    explicit CUSP_spec_impl(const HMM& hmm, size_t level);

    explicit CUSP_spec_impl(size_t level) : precalc_obs_handlers(), level(level){};

    [[nodiscard]] HMM::Mod_prob_vec_t run_Viterbi_spec(const HMM::Emit_seq_t& seq) const override;

    void spec_with(const HMM& hmm) override;

    ~CUSP_spec_impl() override;

  private:
    std::vector<CUSP_helper::Dev_mat> emit_pr_x_start_pr;
    std::vector<CUSP_helper::Dev_mat> emit_pr_x_trans_pr;
    Obs_handler_t precalc_obs_handlers;
    HMM::Index_t states_num;
    size_t level;

    void initializer(const HMM& hmm, size_t level);
    void deleter();
};
