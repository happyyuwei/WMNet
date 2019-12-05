"""
本实验研究嵌入前与嵌入后的差异，从直方图图视觉效果入手
"""

import numpy as np
import matplotlib.pyplot as plt


def show_residual():

    # 载入图片
    image = plt.imread("1_PR.png")[:, :, 0:3]
    image_wm = plt.imread("1_WM.png")[:, :, 0:3]

    image_res = (image-image_wm)+0.7

    image_res[image_res < 0] = 0
    image_res[image_res > 1] = 1

    plt.figure(1)
    plt.imshow(image)

    plt.figure(2)
    plt.imshow(image_wm)

    plt.figure(3)
    plt.imshow(image_res)

    plt.show()

def show_hist():
    image = plt.imread("1_PR.png")[:, :, 0:3]
    image_wm = plt.imread("1_WM.png")[:, :, 0:3]

    image=np.reshape(image, [1,128*128*3])*255
    image_wm=np.reshape(image_wm, [1,128*128*3])*255

    bins=[]
    for i in range(0,51):
        bins.append(5*i)


    print(bins)

    hist,bins = np.histogram(image, bins=bins)  
    hist_wm,bins = np.histogram(image_wm, bins=bins)  

    # plt.figure("normal")
    
    # plt.figure("wm")
    plt.bar(range(len(hist)), hist_wm, color="coral")
    plt.bar(range(len(hist)), hist)
    plt.show()



show_hist()
