#include "cuASR_helper.h"

#include "cuasr/gemm/device/default_srgemm_configuration.h"
#include "cuasr/gemm/device/srgemm.h"
#include "cuasr/functional.h"

#include <cstdlib>
#include <experimental/source_location>
#include <iostream>

namespace {
void check_for_cuda_error(
    [[maybe_unused]] std::experimental::source_location s = std::experimental::source_location::current()) {
#ifndef NDEBUG
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess ) {
        std::cerr << "CUDA Error: " <<  cudaGetErrorString(err) << '\n';
        std::cerr << "    file: " << s.file_name() << '\n'
                  << "    function: " << s.function_name() << '\n'
                  << "    line: " << s.line() << "\n\n";
        std::exit(1);
    }
#endif
}
}


namespace cuASR_helper {

using AdditionOp       = cuasr::minimum<float>;
using MultiplicationOp = cuasr::plus<float>;

using RowMajor = cutlass::layout::RowMajor;

using cuASR_MinPlus_SGEMM = cuasr::gemm::device::Srgemm<
      AdditionOp, MultiplicationOp,
      HMM::Mod_prob_t, RowMajor,
      HMM::Mod_prob_t, RowMajor,
      HMM::Mod_prob_t, RowMajor,
      HMM::Mod_prob_t
      >;

Dev_mat::Dev_mat(HMM::Mod_prob_t* data, int rows, int cols, size_t bytes_size)
    : data(data), rows(rows), cols(cols), bytes_size(bytes_size)
{}

Dev_mat::Dev_mat(const Dev_mat& rhs) {
    rows = rhs.rows;
    cols = rhs.cols;
    bytes_size = rhs.bytes_size;
    cudaMalloc((void **)&data, bytes_size);
    check_for_cuda_error();
    cudaMemcpy(rhs.data, data, bytes_size, cudaMemcpyDeviceToDevice);
    check_for_cuda_error();
}

Dev_mat::~Dev_mat() {
    cudaFree(static_cast<void*>(data));
    check_for_cuda_error();
}

void min_plus_Dev_mat_multiply(const Dev_mat& lhs, const Dev_mat& rhs, Dev_mat& res) {
    if (res.data == nullptr) {
        cudaMalloc((void **)&(res.data), sizeof(HMM::Mod_prob_t) * lhs.rows * rhs.cols);
        check_for_cuda_error();
        res.rows = lhs.rows;
        res.cols = rhs.cols;
    }

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
        {lhs.rows, lhs.cols, rhs.cols},
        {lhs.data, lhs.rows},
        {rhs.data, rhs.rows},
        {res.data, res.rows},
        {res.data, res.rows},
        {MultiplicationOp::Identity, MultiplicationOp::Annihilator}
    );

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
        return {};
    }
#endif
    auto host_data = new HMM::Mod_prob_t[mat.rows];
    cudaMemcpy(host_data, mat.data, mat.bytes_size, cudaMemcpyDeviceToHost);
    check_for_cuda_error();
    auto res = HMM::Mod_prob_vec_t(host_data, host_data + mat.rows);
    delete(host_data);
    return res;
}

void init_matrices_from_HMM(const HMM& hmm, Dev_mat& start_pr, Dev_mat& transp_tr,
    std::vector<Dev_mat>& emit_mat_vec)
{
    // Column for start probs
    auto start_host_ptr = new HMM::Mod_prob_t[hmm.states_num];
    for (size_t i = 0; i < hmm.non_zero_start_probs; ++i) {
        start_host_ptr[hmm.start_probabilities_cols[i]] = hmm.start_probabilities[i];
    }

    // Row major transposed transition matrix
    auto transp_tr_host_ptr = new HMM::Mod_prob_t[hmm.states_num * hmm.states_num];
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
        for (size_t j = 0; j < hmm.states_num; ++j) {
            m[j * hmm.states_num + j] = hmm.emissions[i][j];
        }
    }

    start_pr = Dev_mat {nullptr, (int)hmm.states_num, 1, sizeof(HMM::Mod_prob_t) * hmm.states_num * 1};

    transp_tr = Dev_mat {
        nullptr, (int)hmm.states_num, (int)hmm.states_num, 
        sizeof(HMM::Mod_prob_t) * hmm.states_num * hmm.states_num
    };

    emit_mat_vec = std::vector<Dev_mat>(hmm.emit_num);
    for (size_t i = 0; i < hmm.emit_num; ++i) {
        emit_mat_vec[i] = Dev_mat {
            nullptr, (int)hmm.states_num, (int)hmm.states_num, 
            sizeof(HMM::Mod_prob_t) * hmm.states_num * hmm.states_num
        };
    }


    // Allocate device memory

    cudaMalloc((void **)&start_pr.data, start_pr.bytes_size);
    check_for_cuda_error();
    cudaMalloc((void **)&transp_tr.data, transp_tr.bytes_size);
    check_for_cuda_error();
    for (size_t i = 0; i < hmm.emit_num; ++i) {
        cudaMalloc((void **)&(emit_mat_vec[i].data), emit_mat_vec[i].bytes_size);
        check_for_cuda_error();
    }

    // Transfer data to device
    cudaMemcpy(start_pr.data, start_host_ptr, start_pr.bytes_size, cudaMemcpyHostToDevice);
    check_for_cuda_error();
    cudaMemcpy(transp_tr.data, transp_tr_host_ptr, transp_tr.bytes_size, cudaMemcpyHostToDevice);
    check_for_cuda_error();
    for (size_t i = 0; i < hmm.emit_num; ++i) {
        cudaMemcpy(emit_mat_vec[i].data, emit_mat_vec_host[i], emit_mat_vec[i].bytes_size, cudaMemcpyHostToDevice);
        check_for_cuda_error();
    }

    // Free host memory
    delete(start_host_ptr);
    delete(transp_tr_host_ptr);
    for (auto& m : emit_mat_vec_host) {
        delete(m);
    }
}
} // namespace cuASR_helper
