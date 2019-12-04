"""
随着配置参数越来越多，本人计划重新设计配置文件，先计划如下：
1. 使用json更好管理配置逻辑，现在的键值对key=value方式有些力不从心。
2. 重新设计配置类，使得添加额外配置更方便
2. 增加一个轻量级配置界面，使得每次修改时更加清晰不会出错。
3. 配置接口依旧适用LoadConfig类， 完全兼容旧版本配置逻辑。
4. 在过渡期间将把所有旧应用配置文件转成json,在此期间旧版本配置依旧兼容，但绝对不推荐在新应用中使用旧版本配置，在过渡期后，旧版本配置代码将被移除。
5. 访问设置推荐使用config_loader.config, 该对象为字典，可以直接使用字典方式访问， 如：config_loader.config["dataset"]["buffer_size"]["value"]。
不排除在今后阶段移除写法 config_loader.buffer_size的可能。
@since2019.11.30
@author yuwei
"""

import json

default_config = {

    "keys": ["dataset", "augmentation", "training", "callback"],

    "dataset": {
        "keys": ["buffer_size", "batch_size", "image_size", "input_number"],

        "buffer_size": {
            "value": 0,
            "desc": "载入训练图像时设置的缓存区大小，若设置为0，则一次性载入所有图像。（在训练数据量不大的时候推荐设置为0）"
        },
        "batch_size": {
            "value": 1,
            "desc": "训练时一次输入的图片张数，在生成网络训练中，推荐每次输入一张图像。"
        },
        "image_size": {
            "value": [128, 128, 3],
            "desc": "输入图像的尺寸，依次为：宽、高、通道。数据集中所有图像会裁剪成该尺寸。（注意：多余通道会被直接忽略）。"
        },
        "input_number": {
            "value": 1,
            "desc": "单次输入图片数量，部分应用需要输入多张图片才能输出一张图片。默认输入一张。"
        }
    },
    "augmentation": {
        "keys": ["data_flip", "crop_size"],

        "data_flip": {
            "value": 0,
            "desc": "图像反转概率，在输入图像的时候存在一定的概率翻转图像已达到数据增强的效果。可输入（0-1）的小数。默认为0（不翻转）。"
        },
        "crop_size": {
            "value": [0, 0],
            "desc": "图像裁剪，在输入图像的时候随机裁剪一定的尺寸已达到数据增强。默认情况下关闭。"
        }
    },
    "training": {
        "keys": ["generator", "discriminator", "epoch", "save_period", "data_dir", "checkpoints_dir", "log_dir", "training_device", "remove_history_checkpoints", "load_latest_checkpoints"],
        "generator": {
            "value": "unet",
            "desc": "训练使用的生成器结构，默认使用Unet。包含选择：unet, resnet16。"
        },
        "discriminator": {
            "value": "no",
            "desc": "训练使用的生成器结构，默认不使用判决器。包含选择：no, gan, cgan。"
        },
        "epoch": {
            "value": 2000,
            "desc": "训练轮数。默认为2000轮。"
        },
        "save_period": {
            "value": 1,
            "desc": "保存周期。每过一个保存周期，将会保存训练的检查点以及记录训练日志。"
        },
        "data_dir": {
            "value": "./data/",
            "desc": "训练数据集所在路径，相对位置为当前应用app所在位置。"
        },
        "checkpoints_dir": {
            "value": "./training_checkpoints/",
            "desc": "保存检查点路径，相对位置为当前应用app所在位置。"
        },
        "log_dir": {
            "value": "./log/",
            "desc": "保存训练日志路径，相对位置为当前应用app所在位置。"
        },
        "training_device": {
            "value": "GPU",
            "desc": "训练使用设备，默认使用GPU训练。可选择：CPU，GPU。"
        },
        "remove_history_checkpoints": {
            "value": True,
            "desc": "则只保留最近的训练检查点。若启用该设置，早期检查点均会被移除且无法复原。（注意：若禁用该设置，则所有保存点会被保留，会占用大量磁盘空间。）"
        },
        "load_latest_checkpoints": {
            "value": True,
            "desc": "载入最近的检查点。若关闭该设置，将从头开始训练。"
        }
    },
    "callback": {
        "keys": ["training_callback", "callback_args", "lambda"],
        "training_callback": {
            "value": "train_watermark.InvisibleWMCallback",
            "desc": "设置自定义的回调类，常用与自定义损失函数。若使用默认回调,请输入： default。"
        },
        "callback_args": {
            "value": [],
            "desc": "设置回调函数输入参数，用于初始化自定义的回调参数。输入内容为数组，每个元素请用空格隔开。"
        },
        "lambda": {
            "value": [1, 1],
            "desc": "设置损失函数超参数。输入内容为数组，每个元素请用空格隔开。"
        }
    }
}


def create_config_JSON_temple(config_dir):
    """
    推荐使用JSON格式
    """
    # 生成JSON字符串
    json_str = json.dumps(default_config, ensure_ascii=False, indent=2)
    # 写入文件
    with open(config_dir, "w", encoding="utf-8") as f:
        f.write(json_str)


# 兼容旧版本
INPUT_INTRODUCTION = "#The following are input arguments"
BUFFER_SIZE = "buffer_size"
BATCH_SIZE = "batch_size"
IMAGE_WIDTH = "image_width"
IMAGE_HEIGHT = "image_height"
IMAGE_CHANNEL = "image_channel"
INPUT_NUM = "input_number"
DATA_FLIP = "data_flip"
CROP_WIDTH = "crop_width"
CROP_HEIGHT = "crop_height"
TRAIN_INTRODUCTION = "#The followong are training arguments"
GENERATOR = "generator"
DISCRIMINATOR = "discriminator"
TRAINING_CALLBACK = "training_callback"
CALLBACK_ARGS = "callback_args"
EPOCH = "epoch"
PATCH_SIZE = "patch_size"
SAVE_PERIOD = "save_period"
DATA_DIR = "data_dir"
CHECKPOINTS_DIR = "checkpoints_dir"
LOG_DIR = "log_dir"
TRAINING_DEVICE = "training_device"
REMOVE_HISTORY_CHECKPOINTS = "remove_history_checkpoints"
LOAD_LATEST_CHECKPOINTS = "load_latest_checkpoints"
LAMBDA = "lambda"

SPLIT = "="

"""
旧版本配置文件，目前仍然支持。
等新版本测试完毕，旧版本可能不在支持。
"""


def create_config_temple(config_dir, buffer_size=0, batch_size=1, image_width=128, image_height=128, image_channel=3,
                         input_num=1, data_flip=0, crop_width=0, crop_height=0, training_callback="train_watermark.InvisibleWMCallback",
                         callback_args="noise",
                         generator="unet", discriminator="cgan", epoch=2000,  patch_size=0,
                         save_period=1, data_dir="./data/", checkpoints_dir="./training_checkpoints/",
                         log_dir="./log/", training_device="GPU", remove_history_checkpoints=True,
                         load_latest_checkpoints=True, lamb="1 1"):
    # keys
    config_lines = [BUFFER_SIZE, BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL, INPUT_NUM, DATA_FLIP, CROP_WIDTH,
                    CROP_HEIGHT, TRAINING_CALLBACK, CALLBACK_ARGS, GENERATOR, DISCRIMINATOR, EPOCH, PATCH_SIZE, SAVE_PERIOD, DATA_DIR, CHECKPOINTS_DIR, LOG_DIR, TRAINING_DEVICE,
                    REMOVE_HISTORY_CHECKPOINTS, LOAD_LATEST_CHECKPOINTS, LAMBDA]

    # default values
    value = [buffer_size, batch_size, image_width, image_height, image_channel, input_num, data_flip, crop_width,
             crop_height, training_callback, callback_args, generator, discriminator, epoch, patch_size, save_period, data_dir, checkpoints_dir, log_dir, training_device,
             remove_history_checkpoints, load_latest_checkpoints, lamb]

    for i in range(len(config_lines)):
        config_lines[i] = "{} {} {}\n".format(config_lines[i], SPLIT, value[i])

    #保存
    with open(config_dir, "w") as f:
        f.writelines(config_lines)


class ConfigLoader:

    def __init__(self, config_dir=None):
        """
        @update 2019.11.30
        配置文件2.0， 兼容旧版本。
        @update 2019.11.27
        允许创建空的Loader
        @author yuwei
        @since 2018.12
        :param config_dir:
        """

        # 旧属性兼容
        old_attribate_enable = False

        if config_dir == None:
            # set default params
            self.config = default_config
            # 目前均会启用旧属性兼容
            old_attribate_enable = True
        else:
            # load from file
            """
            如果配置文件是JSON,则调用json解析，否则为旧版本配置文件
            以上做法是为了兼容旧版本，已经不推荐使用旧版本设置配置文件。
            旧版本的配置文件将在今后移除，过度阶段后将不在支持。
            @since 2019.11.30
            @author yuwei
            """
            if ".json" in config_dir:
                with open(config_dir, "r", encoding="utf-8") as f:
                    self.config = json.load(f)

                # 目前均会启用旧属性兼容
                old_attribate_enable = True

            else:
                 # 向下兼容代码
                with open(config_dir, "r") as f:
                    lines = f.readlines()

                # load
                self.config_list = dict()
                for line in lines:
                    # "#"is used to make description
                    # @since 2019.1.18
                    # @version 0.93
                    # @author yuwei
                    if "#" not in line:
                        value = line.strip().split(SPLIT)
                        self.config_list[value[0].strip()] = value[1].strip()

                def get_value(config_list, key, type="string"):
                    """
                    get value from config list and convert to parameter.
                    向下兼容，允许读取旧配置文件0.9。 若有些配置字段在就被配文件中没有出现，则会默认为None，而不会解析报错。
                    @author yuwei
                    @since 2019.9.21
                    :param config_list:
                    :param key:
                    :param type:
                    :return:
                    """
                    try:
                        if type is "string":
                            return config_list[key]
                        elif type is "int":
                            return int(config_list[key])
                        elif type is "float":
                            return float(config_list[key])
                        elif type is "boolean":
                            return config_list[key] in ("True", "true")
                        elif type is "array_float":
                            lambda_temp_array = config_list[key].split(" ")
                            lambda_array = []
                            for i in range(len(lambda_temp_array)):
                                lambda_array.append(
                                    float(lambda_temp_array[i]))
                            return lambda_array
                        elif type is "array_str":
                            return config_list[key].split(" ")
                        else:
                            # if unknown type, just return string
                            return config_list[key]
                    except KeyError:
                        # if the key is not fount, do not throw exception, just return none.
                        return None

                # parse
                # buffer size
                self.buffer_size = get_value(
                    self.config_list, BUFFER_SIZE, type="int")
                # batch size
                self.batch_size = get_value(
                    self.config_list, BATCH_SIZE, type="int")
                # image width
                self.image_width = get_value(
                    self.config_list, IMAGE_WIDTH, type="int")
                # image height
                self.image_height = get_value(
                    self.config_list, IMAGE_HEIGHT, type="int")
                # channel
                self.image_channel = get_value(
                    self.config_list, IMAGE_CHANNEL, type="int")
                # input number
                self.input_num = get_value(
                    self.config_list, INPUT_NUM, type="int")
                # data flip
                self.data_flip = get_value(
                    self.config_list, DATA_FLIP, type="float")
                # crop width
                self.crop_width = get_value(
                    self.config_list, CROP_WIDTH, type="int")
                # crop height
                self.crop_height = get_value(
                    self.config_list, CROP_HEIGHT, type="int")
                # training callback
                self.training_callback = get_value(
                    self.config_list, TRAINING_CALLBACK)
                # callback args
                self.callback_args = get_value(
                    self.config_list, CALLBACK_ARGS, type="array_str")
                # generator
                self.generator = get_value(
                    self.config_list, GENERATOR, type="string")
                # discriminator
                self.discriminator = get_value(
                    self.config_list, DISCRIMINATOR, type="string")
                # epoch
                self.epoch = get_value(self.config_list, EPOCH, type="int")
                # patch size
                self.patch_size = get_value(
                    self.config_list, PATCH_SIZE, type="int")
                # save period
                self.save_period = get_value(
                    self.config_list, SAVE_PERIOD, type="int")
                # data dir
                self.data_dir = get_value(
                    self.config_list, DATA_DIR, type="string")
                # checkpoints
                self.checkpoints_dir = get_value(
                    self.config_list, CHECKPOINTS_DIR, type="string")
                # log dir
                self.log_dir = get_value(
                    self.config_list, LOG_DIR, type="string")
                # training device
                self.training_device = get_value(
                    self.config_list, TRAINING_DEVICE, type="string")
                # remove history
                self.remove_history_checkpoints = get_value(
                    self.config_list, REMOVE_HISTORY_CHECKPOINTS, type="boolean")
                # load latest checkpoint
                self.load_latest_checkpoint = get_value(
                    self.config_list, LOAD_LATEST_CHECKPOINTS, type="boolean")

                # parse lambda array
                self.lambda_array = get_value(
                    self.config_list, LAMBDA, type="array_float")

        if old_attribate_enable == True:
            # 兼容旧版本访问方式，旧属性目前依旧可以访问
            # buffer size
            self.buffer_size = self.config["dataset"]["buffer_size"]["value"]
            # batch size
            self.batch_size = self.config["dataset"]["batch_size"]["value"]
            # image width
            self.image_width = self.config["dataset"]["image_size"]["value"][0]
            # image height
            self.image_height = self.config["dataset"]["image_size"]["value"][1]
            # channel
            self.image_channel = self.config["dataset"]["image_size"]["value"][2]
            # input number
            self.input_num = self.config["dataset"]["input_number"]["value"]
            # data flip
            self.data_flip = self.config["augmentation"]["data_flip"]["value"]
            # crop width
            self.crop_width = self.config["augmentation"]["crop_size"]["value"][0]
            # crop height
            self.crop_height = self.config["augmentation"]["crop_size"]["value"][1]
            # training callback
            self.training_callback = self.config["callback"]["training_callback"]["value"]
            # callback args
            self.callback_args = self.config["callback"]["callback_args"]["value"]
            # parse lambda array
            self.lambda_array = [
                float(x) for x in self.config["callback"]["lambda"]["value"]]
            # generator
            self.generator = self.config["training"]["generator"]["value"]
            # discriminator
            self.discriminator = self.config["training"]["discriminator"]["value"]
            # epoch
            self.epoch = self.config["training"]["epoch"]["value"]
            # save period
            self.save_period = self.config["training"]["save_period"]["value"]
            # data dir
            self.data_dir = self.config["training"]["data_dir"]["value"]
            # checkpoints
            self.checkpoints_dir = self.config["training"]["checkpoints_dir"]["value"]
            # log dir
            self.log_dir = self.config["training"]["log_dir"]["value"]
            # training device
            self.training_device = self.config["training"]["training_device"]["value"]
            # remove history
            self.remove_history_checkpoints = self.config[
                "training"]["remove_history_checkpoints"]["value"]
            # load latest checkpoint
            self.load_latest_checkpoint = self.config["training"]["load_latest_checkpoints"]["value"]
