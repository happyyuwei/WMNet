"""
该文件负责所有的评估策略，在该项目中，主要使用SSMI，psnr与ber(二值)来评价模型
@since 2019.11.27
@author yuwei
"""
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import model_use
import loader
import config
import train_tool


def evaluate(image1, image2, psnr_enable=True, ssim_enable=True, ber_enable=False):
    """
    this is for eval
    可测试psnr, ssim, ner
    :param image1: tensor structure with [1,128,128,3] or [1,32,32,3], the scale is [-1,1]
    :param image2: tensor structure with [1,128,128,3] or [1,32,32,3], the scale is [-1,1]
    :param ssim_enable
    :param psnr_enable
    :param ber_enable
    :return result_set, it is a dir
    """
    image1 = image1*0.5+0.5
    image2 = image2*0.5+0.5

    # result
    result_set = {}

    # invoke
    if psnr_enable == True:
        psnr = tf.image.psnr(image1, image2, max_val=1)
        result_set["psnr"] = np.array(psnr)[0]

    if ssim_enable == True:
        ssim = tf.image.ssim(image1, image2, max_val=1)
        result_set["ssim"] = np.array(ssim)[0]

    if ber_enable == True:
        image1 = np.array(image1[0])
        image2 = np.array(image2[0])
        # filter
        image1[image1 >= 0.5] = 1
        image1[image1 < 0.5] = 0
        image2[image2 >= 0.5] = 1
        image2[image2 < 0.5] = 0
        # ber
        ber = np.mean(np.abs(image1-image2))
        # 先就这么写，把tensor值转成数字
        result_set["ber"] = ber

    # results
    return result_set


def eval_all(data_path, model_path,  visual_result_dir=None, watermark_path=None, watermark_binary=False, decode_path=None, attack_test_func=None):
    """
    评估函数, 评估模型的PSNR与SSIM, 若存在水印，则评估水印的PSNR或BER
    @since 2019.11.27
    @author yuwei
    """

    # create empty config, this is used when loading data
    configs = config.ConfigLoader()
    # load data
    data_loader = loader.DataLoader(data_path, is_training=False)
    dataset = data_loader.load(configs)
    print("load data....")

    # load wateramrk
    watermark_enable = False
    wm_target = None
    if watermark_path != None:
        watermark_enable = True
        """
        @author yuwei
        @update 2019.11.28
        change_scale没有指定，因此读取的水印为[0,1], 与要求[-1,1]不符，使得后续计算误码率有误。现已修复。
        """
        wm_target = train_tool.read_image(
            watermark_path, 32, 32, binary=watermark_binary, change_scale=True)
        print("load watermark....")

    # load model
    print("load model....")
    model = model_use.GeneratorModel(model_path=model_path,
                                     watermark=watermark_enable, decoder_path=decode_path)

    if visual_result_dir != None and os.path.exists(visual_result_dir) == False:
        os.mkdir(visual_result_dir)

    # start eval
    print("start eval....")
    """
    The result set is a key-value dir, 
    which will save 1. mean_value, 2. value_list(the mean value is calculated bu this)
    """
    result_set = {}
    # the value list will save all the results by the image
    value_list = []
    # default value
    image_mean_psnr = 0
    image_mean_ssim = 0
    wm_mean_error = 0
    # image num
    count = 0
    # for each
    for input_image, ground_truth in dataset:
        # result each
        result_each = {}

        # generate
        output_tensor, wm_tensor, wm_feature = model.generate(
            input_image, attack_test_func=attack_test_func)

        # eval image
        image_result_each = evaluate(output_tensor, ground_truth,
                                     psnr_enable=True, ssim_enable=True, ber_enable=False)

        # save results
        result_each["image_psnr"] = image_result_each["psnr"]
        result_each["image_ssim"] = image_result_each["ssim"]

        # caluclate total value
        image_mean_psnr = image_mean_psnr+result_each["image_psnr"]
        image_mean_ssim = image_mean_ssim+result_each["image_ssim"]

        if watermark_path != None:
            # eval watermark
            wm_result_each = evaluate(wm_tensor, wm_target, psnr_enable=(
                not watermark_binary), ssim_enable=False, ber_enable=watermark_binary)

            # calcualte watermark error
            if watermark_binary == True:
                result_each["wm_ber"] = wm_result_each["ber"]
                wm_mean_error = wm_mean_error+result_each["wm_ber"]
            else:
                result_each["wm_psnr"] = wm_result_each["psnr"]
                wm_mean_error = wm_mean_error+result_each["wm_psnr"]

        # append
        value_list.append(result_each)

        # save visual results
        if visual_result_dir != None:

            # basic
            image_list = [input_image, ground_truth, output_tensor]
            title_list = ["IN", "GT", "PR"]

            # wm
            if watermark_path != None:
                image_list.append(wm_tensor)
                image_list.append(wm_feature)
                title_list.append("WM")
                title_list.append("WF")

            # save image
            train_tool.save_images(
                image_list, title_list, visual_result_dir, count+1)

        # one image test finished
        count = count+1
        # this print will flush at the same place
        print("\r"+"testing image {} ...".format(count), end='', flush=True)

    # change line now
    print("")
    # calculate image mean value
    mean_value = {}
    image_mean_psnr = image_mean_psnr/count
    mean_value["psnr"] = image_mean_psnr
    image_mean_ssim = image_mean_ssim/count
    mean_value["ssim"] = image_mean_ssim

    # mean watermark error
    if watermark_path != None:
        wm_mean_error = wm_mean_error/count

        if watermark_binary == True:
            mean_value["wm_ber"] = wm_mean_error
        else:
            mean_value["wm_psnr"] = wm_mean_error

    # save all
    result_set["mean_value"] = mean_value
    result_set["value_list"] = value_list

    # genrate report
    eval_report = "The evaluating report comes here:\n'image psnr' = {}, 'image ssim' = {}".format(
        image_mean_psnr, image_mean_ssim)

    if watermark_path != None:
        eval_report_wm = ", 'watermark {}' = {}"
        if watermark_binary == True:
            wm_format = "ber"
        else:
            wm_format = "psnr"
        eval_report_wm = eval_report_wm.format(
            "wm_"+wm_format, mean_value["wm_"+wm_format])
        eval_report = eval_report+eval_report_wm

    # print(eval_report)
    return result_set, eval_report


def create_noise_attack_func(mean=0, sigma=0.1):
    """
    This function will return a function of creating noise
    @since 2019.11.27
    @author yuwei
    """
    def attack_nosie_test(image):
        normal_noise = np.random.normal(mean, scale=sigma, size=[128, 128, 3])
        # 绘图时会把>1或<0的像素削峰，因此不需再次削峰
        image = image+normal_noise
        return image

    return attack_nosie_test

def create_crop_attack_func(crop_width):
    """
    This function will return a function of creating nrop
    @since 2019.12.6
    @author yuwei
    """
    def attack_crop_test(image):
        # 创建掩码
        crop_mask = np.ones([1, 128, 128, 3], dtype=np.float32)
        # 裁剪长度为0-30个像素宽度
        crop_mask[:, :, 0:crop_width, :] = 0
        # 裁剪
        image = tf.multiply(image, crop_mask)+crop_mask-1

        return image

    return attack_crop_test


def eval_key_sensitive(key_path, input_path, target_path, model_path, visual_result_path=None, key_bits=1024):
    """
    评估密钥敏感性
    """
    # load model
    model = model_use.EncoderDecoder(model_path, key_enable=True)
    print("initial model....")

    # load key in numpy
    secret_key = np.load(key_path)
    # 把他变成长条是为了方便修改
    secret_key = np.reshape(secret_key, [1024, 1])
    print("load model....")

    # input_image = train_tool.read_image(
    #     in, 32, 32, change_scale=True)
    # print("load input....")

    target_image = train_tool.read_image(
        target_path, 32, 32, change_scale=True)
    print("load target....")

    result_list=[]

    # 修改秘钥， 测试改变不同位数下的敏感性
    for i in range(0, key_bits+1):

        # copy key
        temp_key = secret_key.copy()

        # 修改位数，从头开始修改
        for j in range(0, i):
            temp_key[j, 0] = -temp_key[j, 0]

        temp_key=np.reshape(temp_key, [1,32,32,1])

        # save temp key
        np.save("~temp_key.npy", temp_key)



        # decode
        decode_image = model.decode_from_image(
            encode_image_path=input_path, secret_key_path="~temp_key.npy")

        # eval
        decode_image=decode_image*2-1
        decode_image=tf.reshape(decode_image, [1,32,32,3])

        psnr=evaluate(decode_image, target_image, psnr_enable=True, ssim_enable=False, ber_enable=False)

        result_list.append(psnr["psnr"])

        print("\r"+"eval {}..., psnr={}".format(i, psnr["psnr"]), end="", flush=True)

    print()
    # delete temp key file
    os.remove("~temp_key.npy")

    return result_list
        

        

