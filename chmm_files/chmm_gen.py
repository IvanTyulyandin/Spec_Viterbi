import random

# Parameters
states_num: int = 900
trans_per_state: int = 3
transitions_num: int = trans_per_state * states_num
num_non_zero_start_probs: int = 2
emit_range: int = 20

file_name: str = "random_" + \
    str(states_num) + "_" + str(transitions_num) + "_" + \
    str(emit_range) + "_" + str(num_non_zero_start_probs) + ".chmm"

# Implicit parameter for probabilities generation
rng_range: int = 100


def generate_probability_list(length: int) -> list:
    # Fill list with random values, then divide all elements to sum of probs,
    # so sum(probs) == 1
    probs: list = []
    for _ in range(length):
        probs.append(random.randrange(rng_range))
    sum_of_list: int = sum(probs)
    # Cast to floats with fixed precision of 6-2 = 4 signs
    probs = list(
        map(lambda x: str(float(x) / sum_of_list)[:6], probs))
    return probs


# Generation
with open(file_name, 'w') as f:
    f.write(str(states_num) + '\n')

    # Start probabilities pairs info
    start_probs: list = generate_probability_list(num_non_zero_start_probs)
    f.write(str(num_non_zero_start_probs) + '\n')
    for i in range(num_non_zero_start_probs):
        f.write(str(i) + ' ' + start_probs[i] + '\n')

    # Emissions probabilities for each state
    f.write(str(emit_range) + '\n')
    for _ in range(states_num):
        emit_probs: list = generate_probability_list(emit_range)
        emit_str: str = ' '.join(emit_probs) + '\n'
        f.write(emit_str)

    # Transitions info
    f.write(str(transitions_num) + '\n')
    for src in range(states_num):
        used_dst: list = []

        for _ in range(trans_per_state):
            dst: int = random.randrange(states_num)
            while (dst in used_dst):
                dst = random.randrange(states_num)
            used_dst.append(dst)
        trans_probs: list = generate_probability_list(trans_per_state)

        for i in range(trans_per_state):
            f.write(str(src) + ' ' + str(used_dst[i]) +
                    ' ' + trans_probs[i] + '\n')
