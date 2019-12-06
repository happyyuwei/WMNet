import numpy as np
import matplotlib.pyplot as plt

import sys
import getopt

if __name__ == "__main__":

    file = None

    # parse input arguments if exist
    if len(sys.argv) >= 2:
        options, _ = getopt.getopt(sys.argv[1:], "", ["file="])
        print(options)

        for key, value in options:
            if key in ("--file"):
                file = value

    result = np.loadtxt(file)

    for i in range(0, len(result)):
        plt.plot(result[i])

    plt.show()
