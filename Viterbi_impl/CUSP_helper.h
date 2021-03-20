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

HMM::Prob_vec_t Dev_mat_to_Prob_vec(const Dev_mat& mat);

bool is_not_zero_prob(HMM::Probability_t x);

} // namespace CUSP_helper
