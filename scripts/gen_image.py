from scripts.python_utils import unsound, profile
import matplotlib.pyplot as plt

output_folder = "./data/outputs"
test_name = "knownBitsModu"
DEBUG_FILE = "debug.log"
INFO_FILE = "info.log"


def get_debug_file():
    return output_folder + "/" + test_name + "/" + DEBUG_FILE


def get_info_file():
    return output_folder + "/" + test_name + "/" + INFO_FILE


def main():
    debug_file = get_debug_file()
    info_file = get_info_file()
    unsound.plot(debug_file, test_name)
    profile.plot(debug_file)
    plt.show()


main()
