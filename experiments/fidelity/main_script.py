"""
本实验会对比嵌水印和不嵌水印的PSNR, SSIM。分别对比不嵌水印，嵌入二值水印，嵌入彩色水印。
"""
import sys
sys.path.append("../../")

import pickle
import os
import getopt
from evaluate import eval_all



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

    print("\nstart {} testing---------------------------------------------------------------".format(model))
    # evaluate model with watermark
    results, report = eval_all(data_path=dataset,
                               model_path=model,
                               visual_result_dir=visual_result,
                               watermark_path=watermark,
                               watermark_binary=is_binary,
                               decode_path=decoder)

    print("{}---------------------------------------------------------------------------------------".format(model))
    print(report)
    # it seemed not used
    # f = open('./{}/result_wm.pkl'.format(dataset), 'wb')
    # pickle.dump(results, f)
    # f.close()
