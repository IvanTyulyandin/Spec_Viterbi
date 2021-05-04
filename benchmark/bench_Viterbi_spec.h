#include "../Viterbi_impl/CUSP_spec_impl.h"
#include "../Viterbi_impl/GraphBLAS_spec_impl.h"
#include "../Viterbi_impl/data_reader.h"
#include "benchmark_helper.h"

#include <experimental/filesystem>
#include <memory>

namespace benchmark {

namespace fs = std::experimental::filesystem;

using Vec_Viterbi_spec_impls_t = std::vector<std::shared_ptr<Viterbi_spec_impl>>;

void benchmark_Viterbi_spec_impls_to_dat_file(const helper::Folder_path_t& chmm_folder,
                                              const helper::File_name_t& ess_file,
                                              const helper::File_name_t& out_file) {
    const auto ess_seq = read_emit_seq(ess_file);

    // Implementations setup
    GraphBLAS_helper::get_instance().launch_GraphBLAS();

    // Benchmarked implementations
    constexpr auto LEVELS = 3;
    auto impls_to_bench = Vec_Viterbi_spec_impls_t();

    // Headers for .dat file
    auto headers = benchmark::helper::Headers_t({"States"});

    for (size_t i = 1; i <= LEVELS; ++i) {
        impls_to_bench.push_back(std::make_shared<GraphBLAS_spec_impl>(i));
        headers.push_back("GraphBLAS_spec_" + std::to_string(i) + "_prep");
        headers.push_back("GraphBLAS_spec_" + std::to_string(i));
    }

    for (size_t i = 1; i <= LEVELS; ++i) {
        impls_to_bench.push_back(std::make_shared<CUSP_spec_impl>(i));
        headers.push_back("CUSP_spec_" + std::to_string(i) + "_prep");
        headers.push_back("CUSP_spec_" + std::to_string(i));
    }

    auto bench = benchmark::helper::States_time_map();

    for (size_t i = 0; i < impls_to_bench.size(); ++i) {
        std::cout << '\n' << headers[(i + 1) * 2] << " is running!\n";
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

                // Count time to perform specialization
                auto spec_func = [&impl, &hmm = std::as_const(hmm)]() { impl->spec_with(hmm); };
                auto spec_times = benchmark::helper::get_sorted_run_times(spec_func);
                bench[hmm.states_num].push_back(benchmark::helper::get_median(spec_times));

                // Benchmark the impl run time
                const auto func_to_bench = [&impl = std::as_const(impl),
                                            &ess_seq = std::as_const(ess_seq)]() {
                    for (const auto& ess : ess_seq) {
                        // cast to intentionally drop nodiscard
                        static_cast<void>(impl->run_Viterbi_spec(ess));
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
