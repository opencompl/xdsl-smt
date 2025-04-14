import matplotlib.pyplot as plt
import os

PREFIX = "Skip a unsound function at bit width "
PREFIX_LEN = len(PREFIX)


def filter_file(file):
    result = []
    with open(file, "r") as fin:
        for line in fin.readlines():
            if line.startswith(PREFIX):
                result.append(int(line[PREFIX_LEN:]))

    return result


def collect_frequency(data):
    freq = {}
    for ele in data:
        if ele not in freq:
            freq[ele] = 0
        freq[ele] += 1
    return freq


FILE = "./data/kb_modu.log"


def plot(file, test_name):
    data = filter_file(file)
    freq = collect_frequency(data)
    print(freq)
    # data=shrink_data(data,100, max)
    plt.subplot(1, 2, 1)

    bitwidth = list(freq.keys())
    num = list(freq.values())

    plt.bar(bitwidth, num)

    plt.ylabel("Number of unsound functions")
    plt.xlabel("Bitwdith")
    plt.title(os.path.basename(test_name))
