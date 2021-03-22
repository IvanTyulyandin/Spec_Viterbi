#pragma once

#include "Viterbi_spec_impl.h"

#include "CUSP_helper.h"
#include <unordered_map>

class CUSP_spec_impl : public Viterbi_spec_impl {
  public:
    using Obs_handler_t =
        std::unordered_map<std::vector<size_t>, CUSP_helper::Dev_mat, HMM::Emit_seq_hasher>;

    // "level" is the maximum depth for precalc_observation_handlers
    // if level == 1, it makes no sence, same matrices are stored in emit_pr_x_trans_pr
    // if level == 2, it will save all possible (emit_pr_x_trans_pr * emit_pr_x_trans_prit_x_tr)
    // if level == 3: (emit_pr_x_trans_pr * emit_pr_x_trans_pr * emit_pr_x_trans_prt_x_tr)
    // and so on
    explicit CUSP_spec_impl(const HMM& hmm, size_t level);

    explicit CUSP_spec_impl(size_t level) : precalc_obs_handlers(), level(level){};

    [[nodiscard]] HMM::Mod_prob_vec_t run_Viterbi_spec(const HMM::Emit_seq_t& seq) const override;

    size_t get_level() const { return level; }

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
