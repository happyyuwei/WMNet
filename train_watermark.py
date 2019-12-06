"""
由于涉及到的模型与损失函数较多，本人设计新架构减少代码冗余:
所有训练通过配置文件启动train.py,再通过具体训练情况调用该文件夹下的训练过程
@author yuwei
@since 2019.11.23
"""
import os

from model import invisible_extract
from model_use import EncoderDecoder
from config import ArgsParser
import train_tool
import train

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class InvisibleWMCallback:
    """
    This class is for training the invisible watermark
    @since 2019.11.23
    @author yuwei
    """

    def __init__(self, config_loader):
        self.config_loader = config_loader

    def init_process(self, init_params):
        """
        This will be invokde in initial process
        """
        print("This class is for training the invisible watermark.")
        generator_optimizer = init_params["gen_optimizer"]
        generator = init_params["generator"]

        # lambdas in loss
        self.lambda_wm_positive = self.config_loader.lambda_array[0]
        self.lambda_wm_negitive = self.config_loader.lambda_array[1]

        # 解析所有配置
        args_parser = ArgsParser(self.config_loader.callback_args)

        # 解码器
        self.decoder_path = args_parser.get("decoder")

        # training noise attack
        self.noise_attack = args_parser.get("noise") in ("True", "true")
        print("noise attack:{}".format(self.noise_attack))

        # 训练随机裁剪增加水印裁剪攻击能力
        self.crop_attack = args_parser.get("crop") in ("True", "true")
        print("crop attack:{}".format(self.crop_attack))

        # lambdas
        print("lp={}, ln={}".format(
            self.lambda_wm_positive, self.lambda_wm_negitive))

        # init extract network
        print("initial WM extract network")
        self.extractor = invisible_extract.ExtractInvisible()

        # load watermark
        self.watermark_target = train_tool.read_image(
            "./watermark.png", 128, 128, change_scale=True)
        print("load watermark successfully...")

        # create negitive. if no watermark, a 1 matrix will be out
        self.negitive_target = tf.zeros(shape=[1, 128, 128, 3])*1
        print("create negative watermark successfully...")

        # the pretrained encoder-decoder model
        # @update 2019.11.27
        # @author yuwei
        # 修复相对路径bug,否则无法载入模型
        self.decoder_model = EncoderDecoder(self.decoder_path)
        print("load decoder successfully...")

        # checkpoints of extractor
        checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer, generator=generator, extractor=self.extractor)
        print("initial personalized checkpoint....")

        # checkpoint must return
        return checkpoint

    def generator_loss(self, gen_output, target):
        """
        @since 2019.9.12
        @update 2019.11.23
        @author yuwei
        :param gen_output:
        :param target:
        :return:
        """

        # l1 loss, error between the ground truth and the gen output
        l1_loss = tf.reduce_mean(tf.abs(gen_output - target))

        # no attack
        ext_input = gen_output

        # 攻击pipline
        # create watermark
        if self.noise_attack == True:
            # create normal noise sigma from 0-0.4
            sigma = np.random.random()*0.4
            # noise attack， the sigma is 0-0.4, mean is 0
            normal_noise = np.random.normal(0, scale=sigma, size=[128, 128, 3])
            # 添加噪声
            ext_input = gen_output+normal_noise

        if self.crop_attack == True:
            # 创建掩码
            crop_mask = np.ones([1, 128, 128, 3], dtype=np.float32)
            # 裁剪长度为0-50个像素宽度
            crop_width = np.random.randint(0, 50)
            crop_mask[:, :, 0:crop_width, :] = 0
            # 裁剪
            ext_input = tf.multiply(gen_output, crop_mask)+crop_mask-1

        # extract the gen output (with watermark)=>watermark
        extract_watermark = self.extractor(ext_input, training=True)

        # negitive samples
        extract_negitive = self.extractor(target, training=True)

        # watermark error, close to watermark target
        watermark_possitive_loss = tf.reduce_mean(
            tf.abs(self.watermark_target - extract_watermark))

        # negitive error, close to noise (all ones)
        watermark_negitive_loss = tf.reduce_mean(
            tf.abs(self.negitive_target-extract_negitive))

        watermark_total_loss = self.lambda_wm_positive * watermark_possitive_loss + \
            self.lambda_wm_negitive*watermark_negitive_loss
        # total loss
        total_loss = l1_loss + watermark_total_loss

        return total_loss, l1_loss, watermark_possitive_loss, watermark_negitive_loss, watermark_total_loss

    def training_process(self, train_params):
        """
        This will be invoked by the main training process
        """
        input_image = train_params["input_image"]
        target = train_params["target"]
        gen_tape = train_params["gen_tape"]
        generator = train_params["generator"]
        generator_optimizer = train_params["generator_optimizer"]

        # generate image
        gen_output = generator(input_image, training=True)

        # loss
        self.total_gen_loss, self.l1_loss, self.watermark_positive_loss, self.watermark_negitive_loss, self.watermark_total_loss = self.generator_loss(
            gen_output, target)

        # all generator models variables
        generator_variables = generator.variables + self.extractor.variables

        # gradient
        generator_gradients = gen_tape.gradient(
            self.total_gen_loss, generator_variables)
        # optimizer
        generator_optimizer.apply_gradients(
            grads_and_vars=zip(generator_gradients, generator_variables))

    def testing_process(self, test_params):
        """
        This will be invoked in test process
        """

        test_dataset = test_params["test_dataset"]
        generator = test_params["generator"]
        log_tool = test_params["log_tool"]

        # chose one visual result
        for input_image, ground_truth in test_dataset.take(1):
            # test
            # image output
            gen_output = generator(input_image, training=True)

            # extract positive feature
            extract_watermark_feature = self.extractor(
                gen_output, training=True)
            # extract negative feature
            extract_negative_feature = self.extractor(
                ground_truth, training=True)

            # extarct watermark
            extract_watermark = self.decoder_model.decode(
                extract_watermark_feature)
            # extract negative
            extract_negative = self.decoder_model.decode(
                extract_negative_feature)

            image_list = [input_image, ground_truth, gen_output, extract_watermark_feature,
                          extract_negative_feature, extract_watermark, extract_negative]
            title_list = ["IN", "GT", "PR", "WF+", "WF-", "E+", "E-"]
            # save image
            log_tool.save_image_list(image_list, title_list)

        # image ssim and watermark psnr
        image_ssim = 0
        wm_mean_error = 0
        count = 0
        for test_input_image, test_ground_truth in test_dataset:
            # generate
            test_output_image = generator(test_input_image, training=True)
            # calculate image ssim in the test data (all)
            _, ssim = train.eval_psnr_ssim(
                test_output_image, test_ground_truth)

            # calcluate watermark feature me in the test data
            test_extract_watermark = self.extractor(
                test_output_image, training=True)
            wm_mean_error = wm_mean_error + \
                tf.reduce_mean(
                    tf.abs(test_extract_watermark-self.watermark_target))
            image_ssim = image_ssim+ssim[0]
            count = count+1

        # mean
        image_ssim = image_ssim/count
        wm_mean_error = wm_mean_error/count

        # save loss in log file
        log_tool.save_loss(total_gen_loss=self.total_gen_loss, l1_loss=self.l1_loss, wm_positive_loss=self.watermark_positive_loss,
                           wm_negitive_loss=self.watermark_negitive_loss, ssim=image_ssim, wm_error=wm_mean_error)


class VisibleWMCallback:

    """
    This is for training visible watermark
    @since 2019.11.26
    @author yuwei
    """

    def __init__(self, config_loader):
        self.config_loader = config_loader

    def init_process(self, init_params):
        """
        This will be invokde in initial process
        """
        print("This class is for training the visible watermark.")

        # lambdas in loss
        self.lambda_wm = 1

        # load watermark
        self.watermark_target = train_tool.read_image(
            "./watermark.png", 128, 128, binary=True)
        print("load watermark successfully...")

        # create mask
        self.mask = 1 - self.watermark_target
        self.mask = tf.convert_to_tensor(self.mask)
        self.mask = tf.cast(self.mask, tf.float32)
        print("create mask successfully...")
        train_tool.save_image(self.mask, os.path.join(
            self.config_loader.log_dir, "mask.png"))
        print("mask save successfully...")

        # use default checkpoint
        return None

    def generator_loss(self, gen_output, target):
        """
        @since 2019.9.12
        @author yuwei
        :param gen_output:
        :param target:
        :return:
        """
        # mean absolute error
        a = tf.multiply(target, self.mask)
        b = tf.multiply(gen_output, self.mask)
        l1_loss = tf.reduce_mean(tf.abs(a - b))

        #  watermark error
        c = tf.multiply(self.watermark_target, gen_output)
        watermark_loss = tf.reduce_mean(tf.abs(self.watermark_target - c))

        # total loss
        total_loss = l1_loss + self.lambda_wm * watermark_loss

        return total_loss, l1_loss, watermark_loss

    def training_process(self, train_params):
        """
        This will be invoked by the main training process
        """
        input_image = train_params["input_image"]
        target = train_params["target"]
        gen_tape = train_params["gen_tape"]
        generator = train_params["generator"]
        generator_optimizer = train_params["generator_optimizer"]

        # generate image
        gen_output = generator(input_image, training=True)

        # loss
        self.total_loss, self.l1_loss, self.wm_loss = self.generator_loss(
            gen_output, target)

        # gradient
        generator_gradients = gen_tape.gradient(
            self.total_loss, generator.variables)
        # optimizer
        generator_optimizer.apply_gradients(
            grads_and_vars=zip(generator_gradients, generator.variables))

    def testing_process(self, test_params):
        """
        testing process
        """
        test_dataset = test_params["test_dataset"]
        generator = test_params["generator"]
        log_tool = test_params["log_tool"]
        # save test result
        for input_image, ground_truth in test_dataset.take(1):
            log_tool.save_image_plt(
                model=generator, test_input=input_image, tar=ground_truth)
            log_tool.save_image(
                model=generator, test_input=input_image, tar=ground_truth)
            # save loss in log file
            log_tool.save_loss(total_loss=self.total_loss,
                               l1_loss=self.l1_loss, wm_loss=self.wm_loss)
