#include "../Viterbi_impl/cuASR_helper.h"
#include "../Viterbi_impl/cuASR_impl.h"
#include "test_helper.h"

int main() {
    const auto impl = cuASR_impl();
    auto res = 1 - static_cast<int>(test_helper::test_impl(impl));
    std::cout << "CUDA allocs: " << cuASR_helper::Dev_mat::allocs << '\n';
    return res;
}
