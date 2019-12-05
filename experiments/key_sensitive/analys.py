import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    result=np.loadtxt("./test1/result.txt")

    for i in range(0, len(result)):
        plt.plot(result[i])
    
    plt.show()