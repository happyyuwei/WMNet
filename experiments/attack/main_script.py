import sys
sys.path.append("../../")

from evaluate import eval_all, create_noise_attack_func
import tensorflow as tf
import matplotlib.pyplot as plt



if __name__ == "__main__":

    """
    create noise attack function
    """

    
    psnr_list=[]
    sigma_list=[]

    # range from 0, 0.05, 0.1, ..., 0.45, 0.5
    for i in range(0, 11):
        sigma=i*0.05
        # create
        func=create_noise_attack_func(sigma=sigma)
        # eval
        wm_results, wm_report = eval_all(data_path="../../data/flower/train",
                          model_path="../../trained_models/flower_unet_l1_bw",
                          watermark_path="../../watermark/wm_binary.png",visual_result_dir="./results",
                          watermark_binary=True, decode_path="../../trained_models/auto_mnist",
                          attack_test_func=func)
        
        psnr_list.append(wm_results["mean_value"]["wm_ber"])
        sigma_list.append(sigma)
        print("---------------------------------------------------------------------------------------")
        print("sigma={}".format(sigma))
        print(wm_report)
        print("---------------------------------------------------------------------------------------")

    plt.plot(sigma_list,psnr_list)
    plt.show()
        



