#include "../Viterbi_impl/CUSP_spec_impl.h"
#include "test_helper.h"

int main() {
    auto is_test_passed = true;
    for (size_t lvl = 1; lvl <= test_helper::LEVELS_TO_TEST; ++lvl) {
        auto impl = CUSP_spec_impl(lvl);
        is_test_passed &= test_helper::test_spec_impl(impl);
    }
    return 1 - static_cast<int>(is_test_passed);
}
