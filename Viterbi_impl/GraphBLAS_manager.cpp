#include "GraphBLAS_manager.h"

#include <iostream>

bool GraphBLAS_manager::is_GraphBLAS_initialized = false;

void GraphBLAS_manager::launch_GraphBLAS() {
    if (!is_GraphBLAS_initialized) {
        auto info = GrB_init(GrB_Mode::GrB_NONBLOCKING);
        check_for_error(info);
        is_GraphBLAS_initialized = true;
    }
}

GraphBLAS_manager::~GraphBLAS_manager() {
    auto info = GrB_finalize();
    check_for_error(info);
}

void GraphBLAS_manager::check_for_error([[maybe_unused]] const GrB_Info& info) {
#ifndef NDEBUG
    if (!(info == GrB_Info::GrB_SUCCESS || info == GrB_Info::GrB_NO_VALUE)) {
        printf("info: %d error: %s\n", info, GrB_error());
    }
#endif
}

HMM::Prob_vec_t GraphBLAS_manager::GrB_Matrix_to_Prob_vec(GrB_Matrix mat) {
#ifndef NDEBUG
    auto cols = GrB_Index();
    auto deb_info = GrB_Matrix_ncols(&cols, mat);
    check_for_error(deb_info);

    if (cols != 1) {
        std::cerr << "Internal implementation error! GraphBLAS matrix is not a column!\n";
    }
#endif

    auto res_size = GrB_Index();
    auto info = GrB_Matrix_nvals(&res_size, mat);
    check_for_error(info);

    auto data = HMM::Prob_vec_t(res_size);
    auto row_indices = HMM::Index_vec_t(res_size);
    auto col_indices = HMM::Index_vec_t(res_size);

    info = GrB_Matrix_extractTuples_FP32(row_indices.data(), col_indices.data(), data.data(),
                                         &res_size, mat);
    check_for_error(info);

    auto res = HMM::Prob_vec_t(res_size);
    for (size_t i = 0; i < res_size; ++i) {
        res[row_indices[i]] = data[i];
    }

    return res;
}
