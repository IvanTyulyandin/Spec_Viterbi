#include "../Viterbi_impl/GraphBLAS_manager.h"
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
    GraphBLAS_manager::get_instance().launch_GraphBLAS();

    // Benchmarked implementations
    const auto impls_to_bench = Vec_Viterbi_spec_impls_t(
        {std::make_shared<GraphBLAS_spec_impl>(1), std::make_shared<GraphBLAS_spec_impl>(2),
         std::make_shared<GraphBLAS_spec_impl>(3)});

    // Headers for .dat file
    const auto headers = benchmark::helper::Headers_t(
        {"States", "GraphBLAS_spec_1", "GraphBLAS_spec_2", "GraphBLAS_spec_3"});

    auto bench = benchmark::helper::States_time_map();

    for (auto& impl : impls_to_bench) {
        for (const auto& profile : fs::directory_iterator(chmm_folder)) {
            const auto& path = profile.path();
            const auto& chmm_name = path.filename().string();

            if ((path.extension() == ".chmm") && (chmm_name != "test_chmm.chmm")) {
                const auto hmm = read_HMM(path.string());
                impl->spec_with(hmm);

                const auto func_to_bench = [&impl = std::as_const(impl),
                                            &ess_seq = std::as_const(ess_seq)]() {
                    for (const auto& ess : ess_seq) {
                        // cast to intentionally drop nodiscard
                        static_cast<void>(impl->run_Viterbi_spec(ess));
                    }
                };

                auto res_run_times = benchmark::helper::get_sorted_run_times(func_to_bench);
                bench[hmm.states_num].push_back(benchmark::helper::get_median(res_run_times));
            }
        }
    }

    benchmark::helper::print_benchmarks_to_file_as_dat(out_file, headers, bench);
}

} // namespace benchmark
