#include "../Viterbi_impl/CUSP_impl.h"
#include "test_helper.h"

int main() {
    const auto impl = CUSP_impl();
    return 1 - static_cast<int>(test_helper::test_impl(impl));
}
