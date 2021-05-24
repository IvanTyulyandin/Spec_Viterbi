#include "cuASR_helper.h"

#include "cuasr/functional.h"
#include "cuasr/gemm/device/default_srgemm_configuration.h"
#include "cuasr/gemm/device/srgemm.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <experimental/source_location>
#include <iostream>

namespace {
void check_for_cuda_error([[maybe_unused]] std::experimental::source_location s =
                              std::experimental::source_location::current()) {
#ifndef NDEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << '\n';
        std::cerr << "    file: " << s.file_name() << '\n'
                  << "    function: " << s.function_name() << '\n'
                  << "    line: " << s.line() << "\n\n";
        std::exit(1);
    }
#endif
}

void set_to_zero_prob(HMM::Mod_prob_t* data, size_t how_much) {
    for (size_t i = 0; i < how_much; ++i) {
        data[i] = HMM::zero_prob;
    }
}

void cuda_matrix_deleter(cuASR_helper::Dev_mat& mat) {
    if (mat.data != nullptr) {
        cudaFree(static_cast<void*>(mat.data));
        mat.allocs--;
        check_for_cuda_error();
        mat.data = nullptr;
    }
}

void copy_Dev_mat(cuASR_helper::Dev_mat& lhs, const cuASR_helper::Dev_mat& rhs) {
    cuda_matrix_deleter(lhs);
    lhs.rows = rhs.rows;
    lhs.cols = rhs.cols;
    lhs.bytes_size = rhs.bytes_size;
    cudaMalloc((void**)&(lhs.data), lhs.bytes_size);
    check_for_cuda_error();
    lhs.allocs++;
    cudaMemcpy((void*)lhs.data, (const void*)rhs.data, lhs.bytes_size, cudaMemcpyDeviceToDevice);
    check_for_cuda_error();
}

void move_Dev_mat(cuASR_helper::Dev_mat& lhs, cuASR_helper::Dev_mat&& rhs) {
    cuda_matrix_deleter(lhs);
    lhs.rows = rhs.rows;
    lhs.cols = rhs.cols;
    lhs.bytes_size = rhs.bytes_size;
    lhs.data = rhs.data;
    rhs.data = nullptr;
}
} // namespace

namespace cuASR_helper {

int Dev_mat::allocs = 0;

using AdditionOp = cuasr::minimum<float>;
using MultiplicationOp = cuasr::plus<float>;

using RowMajor = cutlass::layout::RowMajor;

using cuASR_MinPlus_SGEMM =
    cuasr::gemm::device::Srgemm<AdditionOp, MultiplicationOp, HMM::Mod_prob_t, RowMajor,
                                HMM::Mod_prob_t, RowMajor, HMM::Mod_prob_t, RowMajor,
                                HMM::Mod_prob_t>;

Dev_mat::Dev_mat(int rows, int cols)
    : rows(rows), cols(cols), bytes_size(rows * cols * sizeof(HMM::Mod_prob_t)) {
    cudaMalloc((void**)&data, bytes_size);
    check_for_cuda_error();
    allocs++;
}

Dev_mat::Dev_mat(const Dev_mat& rhs) : data(nullptr) { copy_Dev_mat(*this, rhs); }

Dev_mat& Dev_mat::operator=(const Dev_mat& rhs) {
    copy_Dev_mat(*this, rhs);
    return *this;
}

Dev_mat::Dev_mat(Dev_mat&& rhs) : data(nullptr) { move_Dev_mat(*this, std::move(rhs)); }

Dev_mat& Dev_mat::operator=(Dev_mat&& rhs) {
    move_Dev_mat(*this, std::move(rhs));
    return *this;
}

Dev_mat::~Dev_mat() { cuda_matrix_deleter(*this); }

void validate_Dev_mat_ptr([[maybe_unused]] const Dev_mat& mat,
                          [[maybe_unused]] const std::string& msg) {
#ifndef NDEBUG
    auto attr = cudaPointerAttributes();
    cudaPointerGetAttributes(&attr, (const void*)mat.data);
    if (attr.memoryType != cudaMemoryTypeDevice) {
        std::cout << "Not a device pointer " << msg << ", is host/unregistered? "
                  << (attr.memoryType == cudaMemoryTypeHost) << ' '
                  << (attr.memoryType == cudaMemoryTypeUnregistered) << '\n';
    } else {
        std::cout << "OK " << msg << '\n';
    }
#endif
}

void min_plus_Dev_mat_multiply(const Dev_mat& lhs, const Dev_mat& rhs, Dev_mat& res) {
    cuda_matrix_deleter(res);
    res = Dev_mat(lhs.rows, rhs.cols);
    validate_Dev_mat_ptr(res, "res");
#ifndef NDEBUG
    if (lhs.cols != rhs.rows) {
        std::cerr << "cuASR: lhs and rhs cols/rows mismatch! "
                  << "Lhs.cols is " << lhs.cols << ". "
                  << "Rhs.cols is " << rhs.rows << '\n';
    }
    if (lhs.rows != res.rows) {
        std::cerr << "cuASR: lhs and res rows mismatch! "
                  << "Lhs.rows is " << lhs.rows << ". "
                  << "Res.rows is " << res.rows << '\n';
    }
    if (rhs.cols != res.cols) {
        std::cerr << "cuASR: rhs and res cols mismatch! "
                  << "Rhs.cols is " << rhs.cols << ". "
                  << "Res.cols is " << res.cols << '\n';
    }
#endif
    auto args = cuASR_MinPlus_SGEMM::Arguments(
        {res.rows, res.cols, lhs.cols}, {lhs.data, lhs.cols}, {rhs.data, rhs.cols},
        {res.data, res.cols}, {res.data, res.cols},
        {MultiplicationOp::Identity, MultiplicationOp::Annihilator});

    auto minplus_gemm = cuASR_MinPlus_SGEMM();
    auto status = minplus_gemm(args, nullptr, nullptr);
    cudaDeviceSynchronize();
    check_for_cuda_error();

    if ((int)status) {
        std::cerr << "Matrix multiply error code " << (int)status << '\n'
                  << cutlassGetStatusString(status) << '\n';
    }
}

HMM::Mod_prob_vec_t Dev_mat_to_Prob_vec(const Dev_mat& mat) {
#ifndef NDEBUG
    if (mat.cols != 1) {
        std::cerr << "Error! cuASR Dev_mat is not a column!\n";
    }
#endif
    auto host_data = new HMM::Mod_prob_t[mat.rows * mat.cols];
    cudaMemcpy((void*)host_data, (const void*)mat.data, mat.bytes_size, cudaMemcpyDeviceToHost);
    check_for_cuda_error();
    auto res = HMM::Mod_prob_vec_t(host_data, host_data + mat.rows * mat.cols);
    std::replace_if(
        res.begin(), res.end(),
        [](auto prob) {
            return HMM::almost_equal(std::numeric_limits<HMM::Mod_prob_t>::max(), prob);
        },
        HMM::zero_prob);
    delete[] host_data;
    return res;
}

void init_matrices_from_HMM(const HMM& hmm, Dev_mat& start_pr, Dev_mat& transp_tr,
                            std::vector<Dev_mat>& emit_mat_vec) {
    // Column for start probs
    auto start_host_ptr = new HMM::Mod_prob_t[hmm.states_num];
    set_to_zero_prob(start_host_ptr, hmm.states_num);
    for (size_t i = 0; i < hmm.non_zero_start_probs; ++i) {
        start_host_ptr[hmm.start_probabilities_cols[i]] = hmm.start_probabilities[i];
    }

    // Row major transposed transition matrix
    auto transp_tr_host_ptr = new HMM::Mod_prob_t[hmm.states_num * hmm.states_num];
    set_to_zero_prob(transp_tr_host_ptr, hmm.states_num * hmm.states_num);
    for (size_t i = 0; i < hmm.trans_num; ++i) {
        auto row = hmm.trans_cols[i];
        auto col = hmm.trans_rows[i];
        auto val = hmm.trans_probs[i];
        transp_tr_host_ptr[row * hmm.states_num + col] = val;
    }

    // Diagonal matrices
    auto emit_mat_vec_host = std::vector<HMM::Mod_prob_t*>(hmm.emit_num);
    for (size_t i = 0; i < hmm.emit_num; ++i) {
        auto& m = emit_mat_vec_host[i];
        m = new HMM::Mod_prob_t[hmm.states_num * hmm.states_num];
        set_to_zero_prob(m, hmm.states_num * hmm.states_num);
        for (size_t j = 0; j < hmm.states_num; ++j) {
            m[j * hmm.states_num + j] = hmm.emissions[i][j];
        }
    }

    start_pr = Dev_mat((int)hmm.states_num, 1);

    transp_tr = Dev_mat((int)hmm.states_num, (int)hmm.states_num);

    emit_mat_vec = std::vector<Dev_mat>(hmm.emit_num);
    for (size_t i = 0; i < hmm.emit_num; ++i) {
        emit_mat_vec[i] = Dev_mat((int)hmm.states_num, (int)hmm.states_num);
    }

    // Transfer data to device
    cudaMemcpy((void*)start_pr.data, (const void*)start_host_ptr, start_pr.bytes_size,
               cudaMemcpyHostToDevice);
    check_for_cuda_error();
    cudaMemcpy((void*)transp_tr.data, (const void*)transp_tr_host_ptr, transp_tr.bytes_size,
               cudaMemcpyHostToDevice);
    check_for_cuda_error();
    for (size_t i = 0; i < hmm.emit_num; ++i) {
        cudaMemcpy((void*)emit_mat_vec[i].data, (const void*)emit_mat_vec_host[i],
                   emit_mat_vec[i].bytes_size, cudaMemcpyHostToDevice);
        check_for_cuda_error();
    }

    // Free host memory
    delete[] start_host_ptr;
    delete[] transp_tr_host_ptr;
    for (auto& m : emit_mat_vec_host) {
        delete[] m;
    }
}
} // namespace cuASR_helper
