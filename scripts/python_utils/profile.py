import matplotlib.pyplot as plt


PREFIX = "Used Time:"


def filter_file(file):
    result = []
    with open(file, "r") as fin:
        for line in fin.readlines():
            if line.startswith(PREFIX):
                result.append(float(line[len(PREFIX) :]))

    return result


def avg(lst):
    return sum(lst) / len(lst)


def shrink_data(data, sub_len, func):
    new_data = []
    tmp = []
    for ele in data:
        tmp.append(ele)
        if len(tmp) == sub_len:
            new_data.append(func(tmp))
            tmp = []
    if len(tmp) != 0:
        new_data.append(func(tmp))
    return new_data


def plot(file):
    data = filter_file(file)
    data = shrink_data(data, 10, max)
    x = [i for i in range(len(data))]
    plt.subplot(1, 2, 2)
    plt.ylabel("Used time(s)")
    plt.xlabel("Steps")
    plt.title("Used time(s)/Steps")
    plt.plot(x, data)
