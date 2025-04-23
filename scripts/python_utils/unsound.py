import matplotlib.pyplot as plt
import os

PREFIX = "Skip a unsound function at bit width "
PREFIX_LEN = len(PREFIX)
PREFIXTF = "Add a new transformer."
PREFIXCTF = "Add a new transformer (cond)."


def filter_file(file):
    result = []
    sound_cnt = 0
    cond_cnt = 0
    with open(file, "r") as fin:
        for line in fin.readlines():
            if line.startswith(PREFIX):
                result.append(int(line[PREFIX_LEN:]))
            elif line.startswith(PREFIXTF):
                sound_cnt += 1
            elif line.startswith(PREFIXCTF):
                cond_cnt += 1
    sound_cnt += cond_cnt
    return result, sound_cnt, cond_cnt


def count_total(file):
    cnt = 0
    with open(file, "r") as fin:
        for line in fin.readlines():
            if line.startswith(PREFIX) or line.startswith(PREFIX2):
                result.append(int(line[PREFIX_LEN:]))


def collect_frequency(data):
    freq = {}
    for ele in data:
        if ele not in freq:
            freq[ele] = 0
        freq[ele] += 1
    return freq


FILE = "./data/kb_modu.log"


def plot(file, test_name):
    data, sound_cnt, cond_cnt = filter_file(file)
    freq = collect_frequency(data)
    print(freq)
    # data=shrink_data(data,100, max)
    plt.subplot(1, 2, 1)

    bitwidth = list(freq.keys())
    num = list(freq.values())

    plt.bar(bitwidth, num)

    plt.text(
        0.95,
        0.95,
        f"Total: {sound_cnt + len(data)}\n Unsound: {len(data)}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.ylabel("Number of unsound functions")
    plt.xlabel("Bitwdith")
    plt.title(os.path.basename(test_name))
