#pragma once

#include <experimental/source_location>

#include "HMM.h"
extern "C" {
#include "GraphBLAS.h"
}

// GraphBLAS can not be initialized and finalized more than once

class GraphBLAS_manager {
  public:
    GraphBLAS_manager(GraphBLAS_manager& other) = delete;
    void operator=(const GraphBLAS_manager&) = delete;

    // Init GraphBLAS
    void launch_GraphBLAS();

    // Finalize GraphBLAS
    ~GraphBLAS_manager();

    static void check_for_error(
        const GrB_Info& info,
        std::experimental::source_location s = std::experimental::source_location::current());

    // Convert from GrB_Matrix to HMM::Prob_vec_t
    // mat expected to be a column
    static HMM::Prob_vec_t GrB_Matrix_to_Prob_vec(GrB_Matrix mat);

    static GraphBLAS_manager& get_instance() {
        static GraphBLAS_manager instance;
        return instance;
    }

  private:
    GraphBLAS_manager() = default;

    static bool is_GraphBLAS_initialized;
};
