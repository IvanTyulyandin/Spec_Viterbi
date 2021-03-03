import random

# Parameters
how_much_to_gen: int = 3
seq_size: int = 7000
emit_range: int = 20
file_name: str = "emit_" + str(how_much_to_gen) + \
    "_" + str(seq_size) + "_" + str(emit_range) + ".ess"
len_per_line: int = 100

# Generation
with open(file_name, 'w') as f:
    f.write(str(how_much_to_gen) + '\n')
    for i in range(how_much_to_gen):
        f.write(str(i) + ' ' + str(seq_size) + '\n')
        random_seq: list = []
        for _ in range(seq_size):
            random_seq.append(str(random.randrange(emit_range)))
        res_seq: list = [" ".join(random_seq[i:i + len_per_line]) + '\n'
                         for i in range(0, seq_size, len_per_line)]
        f.writelines(res_seq)
