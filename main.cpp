#include "benchmark/bench_Viterbi.h"
#include "benchmark/bench_Viterbi_spec.h"

int main() {
    const auto chmm_folder = benchmark::helper::Folder_path_t("chmm_files");
    const auto ess_file = benchmark::helper::File_name_t("ess_files/emit_3_3500_20.ess");
    const auto out_file = benchmark::helper::File_name_t("Viterbi_bench.dat");
    const auto out_file_spec = benchmark::helper::File_name_t("Viterbi_spec_bench.dat");

    // benchmark::benchmark_Viterbi_impls_to_dat_file(chmm_folder, ess_file, out_file);
    benchmark::benchmark_Viterbi_spec_impls_to_dat_file(chmm_folder, ess_file, out_file_spec);
    return 0;
}