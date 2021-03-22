#pragma once

#include <experimental/source_location>

#include "HMM.h"
extern "C" {
#include "GraphBLAS.h"
}

// GraphBLAS can not be initialized and finalized more than once

class GraphBLAS_helper {
  public:
    GraphBLAS_helper(GraphBLAS_helper& other) = delete;
    void operator=(const GraphBLAS_helper&) = delete;

    // Init GraphBLAS
    void launch_GraphBLAS();

    // Finalize GraphBLAS
    ~GraphBLAS_helper();

    static void check_for_error(
        const GrB_Info& info,
        std::experimental::source_location s = std::experimental::source_location::current());

    static void min_plus_mat_multiply(const GrB_Matrix lhs, const GrB_Matrix rhs, GrB_Matrix res);

    // Convert from GrB_Matrix to HMM::Prob_vec_t
    // mat expected to be a column
    static HMM::Mod_prob_vec_t GrB_Matrix_to_Prob_vec(GrB_Matrix mat);

    static GraphBLAS_helper& get_instance() {
        static GraphBLAS_helper instance;
        return instance;
    }

  private:
    GraphBLAS_helper() = default;

    static bool is_GraphBLAS_initialized;
};
