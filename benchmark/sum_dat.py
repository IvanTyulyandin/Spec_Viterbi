import sys

dat_file = sys.argv[1]
with open(dat_file, 'r') as f:
    lines = f.readlines()
    # Skip header
    cols = len(lines[0].split('\t'))
    lines = lines[1:]
    sum_times = [0] * cols
    for line in lines:
        line = line.split('\t')
        for i in range(cols):
            sum_times[i] += int(line[i])
    print(sum_times)
