import sys
sys.path.append("../../")


from evaluate import eval_all
import os
import pickle



if __name__ == "__main__":

    print("start flower_unet_l1_bw testing---------------------------------------------------------------")
    # evaluate model with watermark
    wm_results, wm_report = eval_all(data_path="../../data/flower/train",
                          model_path="../../trained_models/flower_unet_l1_bw",
                          watermark_path="../../watermark/wm_binary.png",
                          watermark_binary=True, decode_path="../../trained_models/auto_mnist")

    print("start flower_unet_l1 testing---------------------------------------------------------------")
    # evaluate model only
    results, report = eval_all(data_path="../../data/flower/train",
                       model_path="../../trained_models/flower_unet_l1")

    print("watermark embedding---------------------------------------------------------------------------------------")
    print(wm_report)
    print("normal---------------------------------------------------------------------------------------")
    print(report)
    
    f = open('./flower/result.txt','wb')
    pickle.dump(results, f)
    f.close()
    f = open('./flower/result_wm.txt','wb')
    pickle.dump(results, f)
    f.close()

