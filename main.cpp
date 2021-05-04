#include "benchmark/bench_Viterbi.h"
#include "benchmark/bench_Viterbi_spec.h"

int main() {
    const auto sequences_datasets = std::vector<std::string>(
        {"emit_3_3500_20", "emit_3_7000_20", "covid-19", "emit_50_3500_20"});
    const auto chmm_folder = benchmark::helper::Folder_path_t("chmm_files");

    for (const auto& dataset_name : sequences_datasets) {
        const auto ess_file = benchmark::helper::File_name_t(std::string("ess_files/") +
                                                             dataset_name + std::string(".ess"));
        const auto out_file = benchmark::helper::File_name_t(std::string("Viterbi_bench_") +
                                                             dataset_name + std::string(".dat"));
        const auto out_file_spec = benchmark::helper::File_name_t(
            std::string("Viterbi_spec_bench_") + dataset_name + std::string(".dat"));

        std::cout << "\n-----------------------------------\n"
                  << dataset_name << " is benchmarking!\n";
        benchmark::benchmark_Viterbi_impls_to_dat_file(chmm_folder, ess_file, out_file);
        benchmark::benchmark_Viterbi_spec_impls_to_dat_file(chmm_folder, ess_file, out_file_spec);
    }
    return 0;
}
