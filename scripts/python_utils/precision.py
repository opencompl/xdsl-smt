import re
import matplotlib.pyplot as plt


def filter_file(file):
    exact_values = []
    dis_values = []
    init_pattern = re.compile(r"^Initial Solution\. Exact: ([\d.]+)%\s+Dis:(\d+)")
    pattern = re.compile(r"^Iter \d+ Finished\. Exact: ([\d.]+)%\s+Dis:(\d+)")
    with open(file, "r") as fin:
        cur_res = None
        for line in fin.readlines():
            init_match = init_pattern.match(line)
            if init_match:
                exact_values.append(float(init_match.group(1)))
                dis_values.append(int(init_match.group(2)))
            match = pattern.match(line)
            if match:
                exact_values.append(float(match.group(1)))
                dis_values.append(int(match.group(2)))
    return exact_values, dis_values


def plot(file):
    exact_data, dis_data = filter_file(file)
    print(exact_data)
    print(dis_data)
    x = list(range(len(exact_data)))

    # Plot with dual y-axis
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left y-axis for Exact%
    ax1.plot(x, exact_data, color="tab:blue", marker="o", label="Exact %")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Exact %", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Right y-axis for Dis
    ax2 = ax1.twinx()
    ax2.plot(x, dis_data, color="tab:red", marker="s", label="Dis")
    ax2.set_ylabel("Dis", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Title and layout
    plt.title("Exact % and Dis over Iterations")
    fig.tight_layout()
    plt.grid(True)
