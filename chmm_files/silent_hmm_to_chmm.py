import math
import sys
import os


def neg_ln_to_prob(prob_inside_ln: float):
    return math.exp(-1 * prob_inside_ln)


def to_fixed(fl: float):
    return "{:.5f}".format(float(fl))


def floats_to_fixed(data: list):
    return map(to_fixed, data)


model_length: int = 0
NUM_OF_AMINO_ACIDS: int = 20
nu: float = 2.0


background_frequencies: list = [
    0.0787945, 0.0151600, 0.0535222, 0.0668298,  # A C D E
    0.0397062, 0.0695071, 0.0229198, 0.0590092,  # F G H I
    0.0594422, 0.0963728, 0.0237718, 0.0414386,  # K L M N
    0.0482904, 0.0395639, 0.0540978, 0.0683364,  # P Q R S
    0.0540687, 0.0673417, 0.0114135, 0.0304133   # T V W Y
]


# Get MSV model from .hmm files
# Probabilities and weights are used interchangeably

for hmm_file in sys.argv[1:]:
    # Trim 4 extension symbols '.hmm'
    hmm_name: str = hmm_file[:-4]
    slash_index: int = hmm_name.rfind('/')
    if slash_index != -1:
        hmm_name = hmm_name[slash_index+1:]

    match_emissions: list = []
    match_transitions: list = []

    with open(hmm_file, 'r') as hmm:
        while True:
            line: str = hmm.readline().lstrip()
            if line.startswith('COMPO'):
                break
            if line.startswith('LENG'):
                model_length = int(line.split()[1])

        # Skip insert emissions line for BEGIN a.k.a. M0 state
        hmm.readline()
        # Read M0 to M1 transition
        # "These seven numbers are: B -> M1, B -> I0, B -> D1; I0 -> M1, I0 -> I"
        match_transitions.append(hmm.readline().split()[0])

        line: str = hmm.readline().strip()
        while line != '//':
            # Parse match emission line, skipping line number
            data: list = line.split()[1:(NUM_OF_AMINO_ACIDS + 1)]
            match_emissions.append(
                list(map(lambda d: neg_ln_to_prob(float(d)), data)))
            # Skip insert emission line for the current node
            hmm.readline()
            # Line is: Mk -> Mk+1, Ik, Dk+1; Ik -> Mk+1, Ik; Dk -> Mk+1, Dk+1
            to_next_match_state = float(hmm.readline().split()[0])
            match_transitions.append(neg_ln_to_prob(to_next_match_state))
            line = hmm.readline().strip()

        # Calculate special states transition probabilities/weights.
        # All non emitting states are merged with emitting states.
        # The J and N states are also merged.
        # Correctness is still a question :)

        # https://github.com/EddyRivasLab/hmmer/blob/master/src/generic_msv.c#L39
        exp_num_of_hits: float = 2.0
        tr_Mk_C: float = (exp_num_of_hits - 1.0) / exp_num_of_hits
        tr_Mk_N: float = 1.0 / exp_num_of_hits

        # Like fs mode in HMMER, page 110 of UserGuide
        # Also https://github.com/EddyRivasLab/hmmer/blob/master/src/generic_msv.c#L144
        tr_move: float = 3 / (model_length + 3)
        tr_loop: float = 1.0 - tr_move

        # I have no idea why this constant is used
        # https://github.com/EddyRivasLab/hmmer/blob/master/src/generic_msv.c#L159
        tr_N_Mk: float = 2.0 / float(model_length * (model_length + 1))

    # start chmm writing
    with open(hmm_name + '.chmm', 'w') as f:
        # All M states + N + C
        # Numbering is N, M1..Mx, C
        states_num = model_length + 2
        f.write(str(states_num) + '\n')
        # Write prob of N state to be start
        f.write('1\n')
        f.write('0 1.0\n')

        # Write emit symbols alphabet and emission probabilities
        alph_cardinality: int = 20
        f.write(str(alph_cardinality) + '\n')

        # Write N state emissions
        f.write(' '.join(floats_to_fixed(background_frequencies)) + '\n')

        # Write M states emissions
        for l in match_emissions:
            f.write(' '.join(floats_to_fixed(l)) + '\n')

        # Write C state emissions
        f.write(' '.join(floats_to_fixed(background_frequencies)) + '\n')

        # Count all transitions: (N) + (M) + (C)
        # From each M_k state: to M_k+1, to N, to C
        # Last M has only 2 transitions: to N and to C
        num_of_trans = (model_length + 1) + (model_length * 3 - 1) + (1)
        f.write(str(num_of_trans) + '\n')

        # Write transitions from N
        f.write('0 0 ' + to_fixed(tr_loop) + '\n')
        for i in range(model_length):
            # Not sure if this is correct
            f.write('0 ' + str(i + 1) + ' ' +
                    to_fixed(tr_move * tr_N_Mk) + '\n')

        # Write transitions for M states
        # From last M there is no transition to the next
        for i in range(model_length - 1):
            # M states are numbered from [1..model_length]
            f.write(str(i + 1) + ' ' + str(i + 2) +
                    ' ' + to_fixed(match_transitions[i]) + '\n')

        for i in range(model_length):
            # From N to M_i
            f.write('0 ' + str(i + 1) + ' ' + to_fixed(tr_N_Mk) + '\n')
            # From M_i to C
            f.write(str(i + 1) + ' ' + str(model_length + 1) +
                    ' ' + to_fixed(tr_Mk_N) + '\n')

        # Transition for C
        f.write(str(model_length + 1) + ' ' +
                str(model_length + 1) + ' ' + to_fixed(tr_loop) + '\n')
