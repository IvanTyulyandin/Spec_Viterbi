#include "data_reader.h"

#include <cmath>
#include <fstream>
#include <iostream>

// HMM format
//
// HMM states amount N
// S — number of states with non zero start probability
// S strings of: "state start_probability"
// E — emit symbols amount
// N strings with E probabilities
// T — transitions amount
// T strings of: "src_state dest_state probability"

HMM read_HMM(const std::string& HMM_file_name) {
    auto file = std::ifstream(HMM_file_name);
    if (file.fail()) {
        std::cerr << "Failed to open file with HMM: " << HMM_file_name << '\n';
        return HMM{};
    }

    auto hmm = HMM{};
    auto prob_from_file = HMM::Probability_t(0);

    // Read number of states
    file >> hmm.states_num;
    hmm.start_probabilities = HMM::Prob_vec_t(hmm.states_num, 0.0);

    // Read states with non zero probability to be start/
    auto state_ind = HMM::Index_t(0);
    auto non_zero_start_prob_num = HMM::Index_t(0);
    file >> non_zero_start_prob_num;

    for (size_t i = 0; i < non_zero_start_prob_num; ++i) {
        file >> state_ind >> prob_from_file;
        prob_from_file = HMM::to_neg_log(prob_from_file);
        hmm.start_probabilities[state_ind] = prob_from_file;
    }

    // Read info about emission symbols
    file >> hmm.emit_num;
    hmm.emissions.reserve(hmm.emit_num * hmm.states_num);

    for (size_t i = 0; i < hmm.emit_num * hmm.states_num; ++i) {
        file >> prob_from_file;
        prob_from_file = HMM::to_neg_log(prob_from_file);
        hmm.emissions.push_back(prob_from_file);
    }

    // Read graph edges info as triples:
    // source_state destination_state transition_probability
    file >> hmm.trans_num;
    hmm.trans_rows = HMM::Index_vec_t();
    hmm.trans_rows.reserve(hmm.trans_num);
    hmm.trans_cols = HMM::Index_vec_t();
    hmm.trans_cols.reserve(hmm.trans_num);

    auto src = size_t(0);
    auto dst = size_t(0);
    for (size_t i = 0; i < hmm.trans_num; ++i) {
        file >> src >> dst >> prob_from_file;
        prob_from_file = HMM::to_neg_log(prob_from_file);
        hmm.trans_rows.push_back(src);
        hmm.trans_cols.push_back(dst);
        hmm.trans_probs.push_back(prob_from_file);
    }

    file.close();
    return hmm;
}

// ess format
// Notes:
//   sequence[_] can be stored at many lines;
//   line with number and lenght should be stored at separate line;
//
// Number of sequences N
// 0 length(sequence[0])
// sequence[0]
// ...
// N-1 length(sequence[N-1])
// sequence[N-1]

HMM::Seq_vec_t read_emit_seq(const std::string& emit_seq_file_name) {
    auto file = std::ifstream(emit_seq_file_name);
    if (file.fail()) {
        std::cerr << "Failed to open file with emitted sequences: " << emit_seq_file_name << '\n';
        return {};
    }

    auto num_of_sequences = size_t(0);
    file >> num_of_sequences;

    auto emitted_sequences = HMM::Seq_vec_t();
    emitted_sequences.reserve(num_of_sequences);

    auto cur_seq = HMM::Emit_vec_t();
    auto cur_seq_num = size_t(0);
    auto cur_seq_len = size_t(0);
    auto cur_emit = HMM::Emit_t(0);

    for (size_t i = 0; i < num_of_sequences; ++i) {
        file >> cur_seq_num;
        if (cur_seq_num != i) {
            std::cerr << "Error in .ess file " << emit_seq_file_name
                      << ": expected sequence number is " << i << ", but read " << cur_seq_num
                      << '\n';
            file.close();
            return {};
        }

        file >> cur_seq_len;
        cur_seq.clear();
        cur_seq.reserve(cur_seq_len);

        for (size_t j = 0; j < cur_seq_len; ++j) {
            file >> cur_emit;
            cur_seq.push_back(cur_emit);
        }
        emitted_sequences.push_back(std::move(cur_seq));
    }

    file.close();
    return emitted_sequences;
}
