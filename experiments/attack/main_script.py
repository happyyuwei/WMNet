import sys
sys.path.append("../../")

from evaluate import eval_all, create_noise_attack_func
import tensorflow as tf
import matplotlib.pyplot as plt
import getopt



if __name__ == "__main__":

    # params
    dataset = None
    model = None
    watermark = None
    is_binary = None
    decoder = None
    visual_result = None

    # parse input arguments if exist
    if len(sys.argv) >= 2:
        options, _ = getopt.getopt(sys.argv[1:], "", [
                                   "data=", "model=", "watermark=", "is_binary=", "decoder=", "visual_result="])
        print(options)

        for key, value in options:
            if key in ("--data"):
                dataset = value
            if key in ("--model"):
                model = value
            if key in ("--watermark"):
                watermark = value
            if key in ("--is_binary"):
                is_binary = (value in ("true", "True"))
            if key in ("--decoder"):
                decoder = value
            if key in ("--visual_result"):
                visual_result = value

    
    if model == None:
        print("Error: No model found...")

    if dataset == None:
        print("Error: No dataset found...")

    if decoder == None:
        print("Error: No decoder found...")    


    psnr_list=[]
    sigma_list=[]


    # range from 0, 0.05, 0.1, ..., 0.45, 0.5
    for i in range(0, 11):
        sigma=i*0.05
        # create
        func=create_noise_attack_func(sigma=sigma)
        # eval
        wm_results, wm_report = eval_all(data_path=dataset,
                          model_path=model,
                          watermark_path=watermark,
                          watermark_binary=is_binary, decode_path=decoder,
                          attack_test_func=func)
        
        psnr_list.append(wm_results["mean_value"]["wm_ber"])
        sigma_list.append(sigma)
        print("---------------------------------------------------------------------------------------")
        print("sigma={}".format(sigma))
        print(wm_report)
        print("---------------------------------------------------------------------------------------")
        
    print(psnr_list)
    plt.plot(sigma_list,psnr_list)
    plt.show()
    
        



