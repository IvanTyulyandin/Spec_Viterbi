#include "../Viterbi_impl/CUSP_impl.h"
#include "../Viterbi_impl/CUSP_spec_impl.h"
#include "../Viterbi_impl/GraphBLAS_impl.h"
#include "../Viterbi_impl/GraphBLAS_spec_impl.h"
#include "test_helper.h"

#include <experimental/filesystem>
#include <memory>

namespace {
constexpr auto chmm_folder = "../chmm_files/";
constexpr auto seq_file_name = "../ess_files/emit_3_3500_20.ess";
constexpr auto SUCCESS = 0;
constexpr auto FAIL = 1;
} // namespace

int main() {
    namespace fs = std::experimental::filesystem;

    auto sequences = read_emit_seq(seq_file_name);
    GraphBLAS_helper::get_instance().launch_GraphBLAS();

    for (const auto& profile : fs::directory_iterator(chmm_folder)) {

        const auto& path = profile.path();
        auto chmm_name = path.filename().string();

        if ((path.extension() == ".chmm") && (chmm_name != "test_chmm.chmm")) {

            auto hmm = read_HMM(path);
            std::cout << chmm_name << '\n';

            // Setup non-specialized implementations
            auto non_spec_impls = std::vector<std::shared_ptr<Viterbi_impl>>();
            non_spec_impls.push_back(std::make_shared<GraphBLAS_impl>());
            non_spec_impls.push_back(std::make_shared<CUSP_impl>());
            auto non_spec_last_answer = HMM::Mod_prob_vec_t();

            // Setup specialized implementations
            auto spec_impls = std::vector<std::shared_ptr<Viterbi_spec_impl>>();
            constexpr size_t MAX_SPEC_LVL = 3;
            for (size_t i = 2; i <= MAX_SPEC_LVL; ++i) {
                spec_impls.push_back(
                    std::shared_ptr<GraphBLAS_spec_impl>(new GraphBLAS_spec_impl(i)));
                spec_impls.back()->spec_with(hmm);

                spec_impls.push_back(std::shared_ptr<CUSP_spec_impl>(new CUSP_spec_impl(i)));
                spec_impls.back()->spec_with(hmm);
            }
            auto spec_last_answer = HMM::Mod_prob_vec_t();

            // Main part of the test
            for (const auto& seq : sequences) {

                // Compare all non-specialized answers
                non_spec_last_answer = non_spec_impls[0]->run_Viterbi(hmm, seq);
                for (size_t i = 1; i < non_spec_impls.size(); ++i) {
                    auto new_answer = non_spec_impls[i]->run_Viterbi(hmm, seq);
                    if (!test_helper::compare_two_answers(non_spec_last_answer, new_answer)) {
                        std::cerr << "Non_spec_impls gave different results!\n"
                                  << i << " and " << i - 1 << '\n';
                        return FAIL;
                    }
                    non_spec_last_answer = std::move(new_answer);
                }

                // Compare all specialized answers
                spec_last_answer = spec_impls[0]->run_Viterbi_spec(seq);
                for (size_t i = 1; i < spec_impls.size(); ++i) {
                    auto new_answer = spec_impls[i]->run_Viterbi_spec(seq);
                    if (!test_helper::compare_two_answers(spec_last_answer, new_answer)) {
                        std::cerr << "Spec_impls gave different results!\n"
                                  << spec_impls[i]->get_level() << ' '
                                  << spec_impls[i - 1]->get_level() << '\n';
                        return FAIL;
                    }
                    spec_last_answer = std::move(new_answer);
                }

                // Compare spec and non_spec answers
                if (!test_helper::compare_two_answers(non_spec_last_answer, spec_last_answer)) {
                    std::cerr << "Non_spec and spec answers are different!\n";
                    return FAIL;
                }
            }
        }
    }

    return SUCCESS;
}
