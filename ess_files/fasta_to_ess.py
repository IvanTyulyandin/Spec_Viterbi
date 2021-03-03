import sys

amino2num = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4',
             'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9',
             'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14',
             'S': '15', 'T': '16', 'V': '17', 'W': '18', 'Y': '19',
             'X': '0'}
# X can be transformed into any aminoacid

for fasta_file in sys.argv[1:]:
    slash_index: int = fasta_file.rfind('/')
    if slash_index != -1:
        ess_file = fasta_file[slash_index+1:]
    else:
        ess_file = fasta_file
    # Trim extension symbols
    ext_pos = ess_file.rfind('.')
    ess_file = ess_file[:ext_pos] + '.ess'

    with open(fasta_file, 'r') as f:
        data = f.readlines()
        data = [x.strip() for x in data]

        seq_list = []
        cur_seq = []
        for line in data:
            if line[0] == '>':
                if cur_seq:
                    seq_list.append(cur_seq)
                cur_seq = []
            else:
                cur_seq.extend(line)
        if cur_seq:
            seq_list.append(cur_seq)

    seq_list = list(map(lambda seq: list(map(
        lambda amino: amino2num[amino], seq)), seq_list))
    seq_sizes = list(map(lambda seq: len(seq), seq_list))

    seq_list = list(map(lambda seq: " ".join(seq), seq_list))

    with open(ess_file, 'w') as f:
        f.write(str(len(seq_list)) + '\n')
        for i, seq in enumerate(seq_list):
            f.write(str(i) + ' ' + str(seq_sizes[i]) + '\n' + seq + '\n')
