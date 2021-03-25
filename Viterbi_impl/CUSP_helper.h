// Select device backend
// THRUST_DEVICE_SYSTEM_XXX
// where XXX can be CUDA, OMP, TBB and CPP
#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP

#include <cusp/coo_matrix.h>

#include "HMM.h"

namespace CUSP_helper {

using Host_mat = cusp::coo_matrix<size_t, HMM::Probability_t, cusp::host_memory>;
using Dev_mat = cusp::coo_matrix<size_t, HMM::Probability_t, cusp::device_memory>;

void min_plus_Dev_mat_multiply(const Dev_mat& lhs, const Dev_mat& rhs, Dev_mat& res);

HMM::Mod_prob_vec_t Dev_mat_to_Prob_vec(const Dev_mat& mat);

void init_matrices_from_HMM(const HMM& hmm, Dev_mat& start_pr_dev, Dev_mat& transp_tr_dev,
                            std::vector<Dev_mat>& emit_mat_vec_dev);

} // namespace CUSP_helper
