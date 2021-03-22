#include "GraphBLAS_helper.h"

#include <iostream>

bool GraphBLAS_helper::is_GraphBLAS_initialized = false;

void GraphBLAS_helper::launch_GraphBLAS() {
    if (!is_GraphBLAS_initialized) {
        auto info = GrB_init(GrB_Mode::GrB_NONBLOCKING);
        check_for_error(info);
        is_GraphBLAS_initialized = true;
    }
}

GraphBLAS_helper::~GraphBLAS_helper() {
    auto info = GrB_finalize();
    check_for_error(info);
}

void GraphBLAS_helper::check_for_error([[maybe_unused]] const GrB_Info& info,
                                       std::experimental::source_location s) {
#ifndef NDEBUG
    if (!(info == GrB_Info::GrB_SUCCESS || info == GrB_Info::GrB_NO_VALUE)) {
        std::cerr << "info: " << info << " error: " << GrB_error() << "Debug info\n"
                  << "    file: " << s.file_name() << '\n'
                  << "    function: " << s.function_name() << '\n'
                  << "    line: " << s.line() << "\n\n";
    }
#endif
}

void GraphBLAS_helper::min_plus_mat_multiply(const GrB_Matrix lhs, const GrB_Matrix rhs,
                                             GrB_Matrix res) {
    auto info = GrB_mxm(res, GrB_NULL, GrB_NULL, GrB_MIN_PLUS_SEMIRING_FP32, lhs, rhs, GrB_NULL);
    GraphBLAS_helper::check_for_error(info);
}

HMM::Mod_prob_vec_t GraphBLAS_helper::GrB_Matrix_to_Prob_vec(GrB_Matrix mat) {
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

    auto data = HMM::Mod_prob_vec_t(res_size);
    auto row_indices = HMM::Index_vec_t(res_size);
    auto col_indices = HMM::Index_vec_t(res_size);

    info = GrB_Matrix_extractTuples_FP32(row_indices.data(), col_indices.data(), data.data(),
                                         &res_size, mat);
    check_for_error(info);

    auto nrows = GrB_Index();
    info = GrB_Matrix_nrows(&nrows, mat);
    check_for_error(info);

    auto res = HMM::Mod_prob_vec_t(nrows, HMM::to_modified_prob(0));
    for (size_t i = 0; i < res_size; ++i) {
        res[row_indices[i]] = data[i];
    }

    return res;
}
