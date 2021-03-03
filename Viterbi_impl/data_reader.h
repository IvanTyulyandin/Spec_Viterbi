#pragma once

#include "HMM.h"

#include <string>

// Return HMM with probabilities stored as negative logarithm
HMM read_HMM(const std::string& HMM_file_name);

// Read emit sequences from .ESS (emitted sequences) file
HMM::Seq_vec_t read_emit_seq(const std::string& emit_seq_file_name);
