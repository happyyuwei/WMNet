"""
该脚本将用于测试不同秘钥的敏感度
@since 2019.11.29
"""
import sys
sys.path.append("../../")

from loader import load_cifar
from model_use import EncoderDecoder
from evaluate import eval_key_sensitive
import numpy as np
import matplotlib.pyplot as plt



def create_encode_images():
    """
    只要用一次
    """
    # data_batch_1 里有1025张图，拿出先200张作为测试
    print("load cifar data....")
    images, _ = load_cifar("../../data/cifar/data_batch_1", 200, 825)

    model=EncoderDecoder("./test4/training_checkpoints", key_enable=True)

    for i in range(0, 200):
        image = np.array(images[i]*0.5+0.5)
        plt.imsave("./test4/data/{}_IN.png".format(i+1),image)
    
    for i in range(0, 200):
        o=model.encode(input_path="./test4/data/{}_IN.png".format(i+1))
        plt.imsave("./test4/data/{}_EN.png".format(i+1),o)




if __name__ == "__main__":

    # create_encode_images()
    

    
    # list = eval_key_sensitive(key_path="../../watermark/key.npy", input_path="./1905_EN.png",
    #                           target_path="./1905_IN.png", model_path="./models/train2")

    # plt.plot(list)
    # ax = plt.subplot(1, 1, 1)
    # ax.plot(list)
    # ax.set_xlabel("error bits in key")
    # ax.set_ylabel("PSNR")
    # plt.show()

    result=[]
    for i in range(1,5):

        list = eval_key_sensitive(key_path="../../watermark/key.npy", input_path="./test4/data/{}_EN.png".format(i),
                              target_path="./test4/data/{}_IN.png".format(i), model_path="./test4/training_checkpoints")
        
        result.append(list)
        np_result=np.array(result)
        #每次都刷新文件
        np.savetxt("./c/result.txt",np_result, "%.2f")
        print("image {} ...".format(i))
        print("-----------------------------------------------------------------------\n")

    

