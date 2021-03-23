#include "../Viterbi_impl/GraphBLAS_helper.h"
#include "../Viterbi_impl/GraphBLAS_impl.h"
#include "test_helper.h"

int main() {

    GraphBLAS_helper::get_instance().launch_GraphBLAS();

    const auto impl = GraphBLAS_impl();
    return 1 - static_cast<int>(test_helper::test_impl(impl));
}
