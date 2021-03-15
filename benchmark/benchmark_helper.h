#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>

namespace benchmark {

namespace helper {

constexpr size_t TIMES_TO_RUN = 2;

using Time_t = size_t;
using Arr_run_times_t = std::array<Time_t, TIMES_TO_RUN>;
using Vec_arr_run_times_t = std::vector<Arr_run_times_t>;

using Headers_t = std::vector<std::string>;
using Benchmark_results = std::vector<benchmark::helper::Time_t>;
using States_time_map = std::map<size_t, Benchmark_results>;

using Folder_path_t = std::string;
using File_name_t = std::string;

namespace {

template <typename T> void print_vector_to_file(std::ofstream& file, const std::vector<T>& vec) {
    for (size_t i = 0; i < vec.size() - 1; ++i) {
        file << vec[i] << '\t';
    }
    file << vec.back() << '\n';
}
} // namespace

Arr_run_times_t get_sorted_run_times(const std::function<void(void)>& func) {
    auto results = Arr_run_times_t();

    for (size_t i = 0; i < TIMES_TO_RUN; ++i) {
        auto iteration_start_time = std::chrono::steady_clock::now();
        func();
        auto cur_time = std::chrono::steady_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - iteration_start_time);
        results[i] = static_cast<Time_t>(duration.count());
    }

    std::sort(results.begin(), results.end());

    return results;
}

// Assumes data is sorted, e.g. after call of get_sorted_run_times
Time_t get_median(const Arr_run_times_t& data) {
    static_assert(TIMES_TO_RUN > 1);
    constexpr auto mid = TIMES_TO_RUN / 2;
    if constexpr (TIMES_TO_RUN % 2) {
        return (data[mid]);
    } else {
        return ((data[mid - 1] + data[mid]) / 2);
    }
}

void print_benchmarks_to_file_as_dat(const File_name_t& file_name, const Headers_t& headers,
                                     const States_time_map& data) {
    // First header is "States"
    auto expected_size = headers.size() - 1;

    auto file = std::ofstream(file_name);
    if (file.fail()) {
        std::cerr << "Failed to open output file: " << file_name << '\n';
        return;
    }

    print_vector_to_file(file, headers);

    for (const auto& [k, v] : data) {
        if (v.size() != expected_size) {
            std::cerr << "Error! Headers size is incompatible with benchmarks, key is " << k
                      << ", values size is " << v.size() << '\n';
        }
        file << k << '\t';
        print_vector_to_file(file, v);
    }
}

} // namespace helper
} // namespace benchmark
