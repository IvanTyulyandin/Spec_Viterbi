#include "../Viterbi_impl/CUSP_impl.h"
#include "../Viterbi_impl/GraphBLAS_helper.h"
#include "../Viterbi_impl/GraphBLAS_impl.h"
#include "../Viterbi_impl/cuASR_impl.h"
#include "../Viterbi_impl/data_reader.h"
#include "benchmark_helper.h"

#include <experimental/filesystem>
#include <memory>

namespace benchmark {

namespace fs = std::experimental::filesystem;

using Vec_Viterbi_impls_t = std::vector<std::shared_ptr<Viterbi_impl>>;

void benchmark_Viterbi_impls_to_dat_file(const helper::Folder_path_t& chmm_folder,
                                         const helper::File_name_t& ess_file,
                                         const helper::File_name_t& out_file) {
    const auto ess_seq = read_emit_seq(ess_file);

    // Implementations setup
    GraphBLAS_helper::get_instance().launch_GraphBLAS();

    // Benchmarked implementations
    const auto impls_to_bench =
        Vec_Viterbi_impls_t({std::make_shared<GraphBLAS_impl>(), std::make_shared<CUSP_impl>(),
                             std::make_shared<cuASR_impl>()});

    // Headers for .dat file
    const auto headers = benchmark::helper::Headers_t({"States", "GraphBLAS", "CUSP", "cuASR"});

    auto bench = benchmark::helper::States_time_map();

    for (size_t i = 0; i < impls_to_bench.size(); ++i) {
        std::cout << '\n' << headers[i + 1] << " is running!\n";
        for (const auto& profile : fs::directory_iterator(chmm_folder)) {
            const auto& path = profile.path();
            const auto& chmm_name = path.filename().string();

#ifndef NDEBUG
            if ((chmm_name != "100.chmm") && (chmm_name != "200.chmm")) {
                std::cout << "Skip " << chmm_name << '\n';
                continue;
            }
#endif

            if ((path.extension() == ".chmm") && (chmm_name != "test_chmm.chmm")) {
                const auto hmm = read_HMM(path.string());
                auto impl = impls_to_bench[i];
                const auto func_to_bench = [&impl = std::as_const(impl), &hmm = std::as_const(hmm),
                                            &ess_seq = std::as_const(ess_seq)]() {
                    for (const auto& ess : ess_seq) {
                        // cast to intentionally drop nodiscard
                        static_cast<void>(impl->run_Viterbi(hmm, ess));
                    }
                };

                auto res_run_times = benchmark::helper::get_sorted_run_times(func_to_bench);
                bench[hmm.states_num].push_back(benchmark::helper::get_median(res_run_times));
                std::cout << chmm_name << " was benchmarked!\n";
            }
        }
    }

    benchmark::helper::print_benchmarks_to_file_as_dat(out_file, headers, bench);
}

} // namespace benchmark
