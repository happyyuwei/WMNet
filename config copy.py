# key
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
CALLBACK_ARGS="callback_args"
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

    with open(config_dir, "w") as f:
        f.writelines(config_lines)


class ConfigLoader:

    def __init__(self, config_dir=None):
        """
        @update 2019.11.27
        允许创建空的Loader
        @author yuwei
        @since 2018.12
        :param config_dir:
        """
        if config_dir==None:
            # load some default params
            self.buffer_size=0
            self.input_num=1
            self.image_width=128
            self.image_height=128
            self.data_flip=0
            self.crop_width=0
            self.crop_height=0
            self.batch_size=1
        else:
            # load from file
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
                向下兼容，允许读取旧配置文件
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
                            lambda_array.append(float(lambda_temp_array[i]))
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
            self.buffer_size = get_value(self.config_list, BUFFER_SIZE, type="int")
            # batch size
            self.batch_size = get_value(self.config_list, BATCH_SIZE, type="int")
            # image width
            self.image_width = get_value(self.config_list, IMAGE_WIDTH, type="int")
            # image height
            self.image_height = get_value(
                self.config_list, IMAGE_HEIGHT, type="int")
            # channel
            self.image_channel = get_value(
                self.config_list, IMAGE_CHANNEL, type="int")
            # input number
            self.input_num = get_value(self.config_list, INPUT_NUM, type="int")
            # data flip
            self.data_flip = get_value(self.config_list, DATA_FLIP, type="float")
            # crop width
            self.crop_width = get_value(self.config_list, CROP_WIDTH, type="int")
            # crop height
            self.crop_height = get_value(self.config_list, CROP_HEIGHT, type="int")
            #training callback
            self.training_callback=get_value(self.config_list, TRAINING_CALLBACK)
            #callback args
            self.callback_args = get_value(self.config_list, CALLBACK_ARGS, type="array_str")
            # generator
            self.generator = get_value(self.config_list, GENERATOR, type="string")
            # discriminator
            self.discriminator = get_value(
                self.config_list, DISCRIMINATOR, type="string")
            # epoch
            self.epoch = get_value(self.config_list, EPOCH, type="int")
            # patch size
            self.patch_size = get_value(self.config_list, PATCH_SIZE, type="int")
            # save period
            self.save_period = get_value(self.config_list, SAVE_PERIOD, type="int")
            # data dir
            self.data_dir = get_value(self.config_list, DATA_DIR, type="string")
            # checkpoints
            self.checkpoints_dir = get_value(
                self.config_list, CHECKPOINTS_DIR, type="string")
            # log dir
            self.log_dir = get_value(self.config_list, LOG_DIR, type="string")
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
            self.lambda_array = get_value(self.config_list, LAMBDA, type="array_float")



