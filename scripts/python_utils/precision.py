import matplotlib.pyplot as plt


PREFIX = "Add a new transformer. Exact: "
PREFIX_B = "Add a existing transformer. Exact: "
PREFIX_LEN = len(PREFIX)
PREFIX_B_LEN = len(PREFIX_B)
NEW_ITER = "Iter "


def filter_file(file):
    result = []
    with open(file, "r") as fin:
        cur_res = None
        for line in fin.readlines():
            if line.startswith(PREFIX):
                cur_res = float(line[PREFIX_LEN : PREFIX_LEN + 5])
            if line.startswith(PREFIX_B):
                cur_res = float(line[PREFIX_B_LEN : PREFIX_B_LEN + 5])
            if line.startswith(NEW_ITER):
                result.append(cur_res)
                cur_res = None
    return result[1:]


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
    print(data)
    # data=shrink_data(data,100, max)
    x = [i for i in range(len(data))]
    plt.plot(x, data)
