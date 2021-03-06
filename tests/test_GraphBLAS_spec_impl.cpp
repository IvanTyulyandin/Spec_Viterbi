#include "../Viterbi_impl/GraphBLAS_helper.h"
#include "../Viterbi_impl/GraphBLAS_spec_impl.h"
#include "test_helper.h"

int main() {

    GraphBLAS_helper::get_instance().launch_GraphBLAS();

    auto is_test_passed = true;
    for (size_t lvl = 1; lvl <= test_helper::LEVELS_TO_TEST; ++lvl) {
        auto impl = GraphBLAS_spec_impl(lvl);
        is_test_passed &= test_helper::test_spec_impl(impl);
    }
    return 1 - static_cast<int>(is_test_passed);
}
