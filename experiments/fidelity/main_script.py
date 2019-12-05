"""
本实验会对比嵌水印和不嵌水印的PSNR, SSIM。
"""
import sys
sys.path.append("../../")

from evaluate import eval_all
import getopt
import os
import pickle



if __name__ == "__main__":

    # params
    dataset = None
    model = None
    wm_model = None
    watermark = None
    is_binary = None
    decoder = None
    visual_result = None
    wm_visual_result = None

    # parse input arguments if exist
    if len(sys.argv) >= 2:
        options, _ = getopt.getopt(sys.argv[1:], "", [
                                   "data=", "model=", "watermark=", "is_binary=", "decoder=", "wm_model=", "visual_result=", "wm_visual_result="])
        print(options)

        for key, value in options:
            if key in ("--data"):
                dataset = value
            if key in ("--model"):
                model = value
            if key in ("--wm_model"):
                wm_model = value
            if key in ("--watermark"):
                watermark = value
            if key in ("--is_binary"):
                is_binary = (value in ("true", "True"))

            if key in ("--decoder"):
                decoder = value

            if key in ("--visual_result"):
                visual_result = value

            if key in ("--wm_visual_result"):
                wm_visual_result = value

    # 只有输入水印模型时，才会进行该水印部分评估
    if wm_model != None:

        print("start {} testing---------------------------------------------------------------".format(wm_model))
        # evaluate model with watermark
        wm_results, wm_report = eval_all(data_path=dataset,
                                         model_path=wm_model,
                                         visual_result_dir=wm_visual_result,
                                         watermark_path=watermark,
                                         watermark_binary=is_binary,
                                         decode_path=decoder)

    if model != None:
        print("start {} testing---------------------------------------------------------------".format(model))
        # evaluate model only
        results, report = eval_all(data_path=dataset,
                                   visual_result_dir=visual_result,
                                   model_path=model)

    if wm_model != None:
        print("watermark embedding---------------------------------------------------------------------------------------")
        print(wm_report)
        f = open('./{}/result.pkl'.format(dataset), 'wb')
        pickle.dump(results, f)
        f.close()

    if model != None:
        print("normal---------------------------------------------------------------------------------------")
        print(report)
        f = open('./{}/result_wm.pkl'.format(dataset), 'wb')
        pickle.dump(results, f)
        f.close()
