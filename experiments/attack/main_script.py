import sys
sys.path.append("../../")

import getopt
import matplotlib.pyplot as plt
import tensorflow as tf
from evaluate import eval_all, create_noise_attack_func, create_crop_attack_func



if __name__ == "__main__":

    # params
    dataset = None
    model = None
    watermark = None
    is_binary = None
    decoder = None
    visual_result = None
    attack = None

    # parse input arguments if exist
    if len(sys.argv) >= 2:
        options, _ = getopt.getopt(sys.argv[1:], "", [
                                   "data=", "model=", "watermark=", "is_binary=", "decoder=", "visual_result=", "attack="])
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
            if key in ("--attack"):
                attack = value

    if model == None:
        print("Error: No model found...")
        sys.exit()

    if dataset == None:
        print("Error: No dataset found...")
        sys.exit()

    if decoder == None:
        print("Error: No decoder found...")
        sys.exit()

    print("attack={}".format(attack))

    y_list = []
    x_list = []

    if is_binary == True:
        value = "wm_ber"
    else:
        value = "wm_psnr"

    # range from 0, 0.05, 0.1, ..., 0.45, 0.5
    for i in range(0, 11):

        # create attack
        if attack == "noise":
            x = i*0.05
            func = create_noise_attack_func(sigma=x)
            x_list.append(x)
        elif attack == "crop":
            x = i*5
            func = create_crop_attack_func(crop_width=x)
            x_list.append(x)

        # eval
        wm_results, wm_report = eval_all(data_path=dataset,
                                         model_path=model,
                                         watermark_path=watermark,
                                         watermark_binary=is_binary, decode_path=decoder,
                                         attack_test_func=func, visual_result_dir="f")

        y_list.append(wm_results["mean_value"][value])
        print("---------------------------------------------------------------------------------------")
        print("x={}".format(x))
        print(wm_report)
        print("---------------------------------------------------------------------------------------")

    print(y_list)
    plt.plot(x_list, y_list)
    plt.show()
