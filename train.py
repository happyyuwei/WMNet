import loader
import config
import train_tool
from model import cgan
import getopt
import sys
import time
import os
import importlib
import numpy as np
import tensorflow as tf

#向下兼容
# tensorflow 2.0 默认开启动态图，并且已经移除该api
if int(tf.__version__.split(".")[0]) <=1:
    tf.enable_eager_execution()



"""
@author yuwei
The load image function is removed to loader.py file, in which one or two input are both supported.
But more than 2 input are not supported, since currently, we don't think more than 2 input are necessary.
Maybe one day, 3 or more inputs will be used, and i will update the code anyway,  but not today.
"""
# finally delete these codes at @since 2019.1.24
"""
i don't know other situations, but to be honest, patch-gan is meanless in generating chinese charachers.
it is difficult to generate words and make me falling many time.
Do not trust anything totally, be criticized.
"""

"""
that's funny, i tried the patch gan before, but not very useful. today, i want to try again,
luckly, i didn't delete these codes, so i don't have to write it again.
maybe, it is still useful, i don;t know anyway
@since 2019.1.24. 22:30
"""

"""
current support models in generator and discriminator
"""
generator_list = ["unet", "resnet16"]
# no means no use discriminator
discriminator_list = ["cgan", "gan", "no"]

"""
@since 2019.11.26
@author yuwei
若为自定义训练，则callback类中需要包含以下方法：
"""
INIT_PROCESS = "init_process"
TRAINING_PROCESS = "training_process"
TESTING_PROCESS = "testing_process"


def patch(image, patch_size):
    """
    split the image to small patches
    :param image:
    :param patch_size:
    :return:
    """

    # get image size and calculate patch
    size = np.array(tf.shape(image))
    row_num = int(size[1] / patch_size)
    col_num = int(size[2] / patch_size)
    # patch list
    patch_list = []

    # split patch
    for i in range(row_num):
        for j in range(col_num):
            patch = image[:, patch_size * i:patch_size *
                          (i + 1), patch_size * j:patch_size * (j + 1), :]
            patch_list.append(patch)

    return patch_list


"""
training iterator is from here, including global discriminator and local discriminator.
i don'y know whether it will be useful, just try.
@ since 2019.1.24
@ author yuwei
@ version 0.94
"""


def global_train_iterator(input_image, target, gen_tape, global_disc_tape,
                          generator, discriminator, generator_optimizer, discriminator_optimizer):
    """
    global train iterator, move to here, before @since 2019.1.24, it is in the train function body
    but the optimizer is become more and more and move to a specific function will be more clear
    @ author yuwei
    @ version 0.94 deep with global and local discriminator
    @ since 2019.1.24
    :param input_image:
    :param target:
    :param gen_tape:
    :param global_disc_tape:
    :param generator:
    :param discriminator:
    :param generator_optimizer:
    :param discriminator_optimizer:
    :return:
    """
    # discriminator is not used
    if discriminator == None:
        # generator output
        gen_output = generator(input_image, training=True)
        # generator loss
        gen_loss = cgan.generator_loss(
            disc_generated_output=None, gen_output=gen_output, target=target)

        # generator tap
        generator_gradients = gen_tape.gradient(gen_loss, generator.variables)

        # generator optimize
        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.variables))

        return gen_loss, None

    # discriminator is used
    else:
        # generator output
        gen_output = generator(input_image, training=True)
        # discriminator output
        # real
        disc_real_output = discriminator(input_image, target, training=True)
        # fake
        disc_generated_output = discriminator(
            input_image, gen_output, training=True)
        # generator loss
        gen_loss = cgan.generator_loss(
            disc_generated_output=disc_generated_output, gen_output=gen_output, target=target)
        # discriminator loss
        disc_loss = cgan.discriminator_loss(
            disc_real_output, disc_generated_output)

        # generator tap
        generator_gradients = gen_tape.gradient(gen_loss, generator.variables)
        # discriminator tap
        discriminator_gradients = global_disc_tape.gradient(
            disc_loss, discriminator.variables)

        # generator optimize
        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.variables))
        # discriminator optimize
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.variables))

        return gen_loss, disc_loss


def eval_psnr_ssim(image1, image2):
    """
    @since 2019.11.21
    @author yuwei
    calculate psnr and ssim in tensor
    """

    # scale the image to 0-255
    image1 = image1+1
    image1 = image1*127.5
    image2 = image2+1
    image2 = image2*127.5

    # invoke
    psnr = tf.image.psnr(image1, image2, max_val=255)
    ssim = tf.image.ssim(image1, image2, max_val=255)
    return psnr, ssim


def eval_psnr_ssim_forall(test_dataset, generator):
    """
    rubbish type
    @since 2019.11.23
    @author yuwei
    """
    # evaluate psnr and ssim in all the test data
    total_psnr = 0
    total_ssim = 0
    count = 0
    for test_input_image, test_ground_truth in test_dataset:
        # generate
        test_output_image = generator(test_input_image, training=True)
        # calculate psnr and ssim in the test data (all)
        psnr, ssim = eval_psnr_ssim(test_output_image, test_ground_truth)
        total_psnr = total_psnr+psnr[0]
        total_ssim = total_ssim+ssim[0]
        count = count+1

    return total_psnr/count, total_ssim/count


def package_params(**kwargs):
    """
    package all these params in a directionary
    this is used when invoking personized training progress
    @author yuwei
    @since 2019.11.23
    """
    params = {}

    for key in kwargs:
        if(kwargs[key] != None):
            params[key] = kwargs[key]

    return params


def invoke_method_of_class(class_instance, method_name, params):
    """
    反射调用实例方法,若方法不存在,则忽略
    @author yuwei
    @since 2019.11.26
    """
    try:
        # get method
        method = getattr(class_instance, method_name)
        # invoke
        return method(params)
    except AttributeError:
        return None
    except Exception as e:
        """
        @since 3019.12.6
        @author yuwei
        修复Bug: 修复调用函数中遇到的任何异常均会被忽略的问题
        """
        print(e)
        sys.exit()


def train(train_dataset, test_dataset, config_loader, training_callback=None):
    """
    train main loop
    :param train_dataset:
    :param test_dataset:
    :param config_loader:
    :param training_function
    :return:
    """
    # start
    print("initial training progress....")

    # checkpoint instance
    checkpoint_prefix = os.path.join(config_loader.checkpoints_dir, "ckpt")

    # phase use discriminator
    use_discriminator = True
    if config_loader.discriminator == "no":
        use_discriminator = False
    print("use discriminator:{}".format(use_discriminator))

    # The call function of Generator and Discriminator have been decorated
    # with tf.contrib.eager.defun()
    # We get a performance speedup if defun is used (~25 seconds per epoch)
    print("initial generator....")
    generator = cgan.Generator()

    global_discriminator = None
    if use_discriminator == True:
        print("initial global discriminator....")
        global_discriminator = cgan.Discriminator()

    # initial optimizer
    print("initial generator optimizer....")
    generator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)

    global_discriminator_optimizer = None
    if use_discriminator == True:
        print("initial global discriminator optimizer....")
        global_discriminator_optimizer = tf.train.AdamOptimizer(
            2e-4, beta1=0.5)

     # init train log
    print("initial train log....")
    log_tool = train_tool.LogTool(
        config_loader.log_dir, config_loader.save_period)

    # initial checkpoint
    checkpoint = None
    """
    由于tensorflow的Checkpoint设计如此失败，不能动态保存模型，
    因此在调用回调时，必须显示创建而无法交给该脚本完成。
    @author yuwei
    @since 2019.11.26
    """
    # init train callback
    if(training_callback != None):
        print("initial personalized callback....")
        # package params
        init_param = package_params(log_tool=log_tool, generator=generator, discriminator=global_discriminator,
                                    gen_optimizer=generator_optimizer, disc_optimizer=global_discriminator_optimizer)
        checkpoint = invoke_method_of_class(
            training_callback, INIT_PROCESS, init_param)
    else:
        print("initial default callback....")

    # 如果checkpoint没有指定返回，均使用默认检查点
    if checkpoint == None:
        # use default checkpoints
        if use_discriminator == True:
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                             global_discriminator_optimizer=global_discriminator_optimizer,
                                             generator=generator,
                                             global_discriminator=global_discriminator)
        else:
            checkpoint = tf.train.Checkpoint(
                generator_optimizer=generator_optimizer, generator=generator)
        print("initial default checkpoint....")

    # restoring the latest checkpoint in checkpoint_dir if necessary
    if config_loader.load_latest_checkpoint == True:
        checkpoint.restore(tf.train.latest_checkpoint(
            config_loader.checkpoints_dir))
        print("load latest checkpoint....")

    def train_each_round(epoch):
        # initial input number
        image_num = 1
        # initial loss
        gen_loss = 0
        disc_loss = 0
        # each epoch, run all the images
        for input_image, target in train_dataset:
            # @since 2019.11.28 将该打印流变成原地刷新
            print("\r"+"input_image {}...".format(image_num), end="", flush=True)
            # calculate input number
            image_num = image_num + 1

            with tf.GradientTape() as gen_tape, tf.GradientTape() as global_disc_tape:
                # default train process
                if training_callback == None:
                    # global train process
                    gen_loss, disc_loss = global_train_iterator(input_image, target, gen_tape, global_disc_tape, generator,
                                                                global_discriminator, generator_optimizer,
                                                                global_discriminator_optimizer)
                else:
                    # package the params
                    train_params = package_params(input_image=input_image, target=target, gen_tape=gen_tape, global_disc_tape=global_disc_tape, generator=generator, log_tool=log_tool,
                                                  global_discriminator=global_discriminator, generator_optimizer=generator_optimizer, global_discriminator_optimizer=global_discriminator_optimizer)
                    # invoke personalized process
                    invoke_method_of_class(
                        training_callback, TRAINING_PROCESS, train_params)

        # change line
        print()
        # save test result
        if epoch % config_loader.save_period == 0:
            # default callback
            if training_callback == None:
                # draw one visual result in the test data
                for test_input_image, test_ground_truth in test_dataset.take(1):
                    # save visible images results
                    log_tool.save_image_plt(
                        model=generator, test_input=test_input_image, tar=test_ground_truth)
                    log_tool.save_image(
                        model=generator, test_input=test_input_image, tar=test_ground_truth)

                # evaluate psnr and ssim in all the test data
                mean_psnr, mean_ssim = eval_psnr_ssim_forall(
                    test_dataset, generator=generator)
                # save values
                log_tool.save_loss(
                    gen_loss=gen_loss, disc_loss=disc_loss, psnr=mean_psnr, ssim=mean_ssim)

            # personalized callback
            else:
                # package the params
                test_params = package_params(
                    test_dataset=test_dataset, log_tool=log_tool, generator=generator)
                # invoke personalized process
                invoke_method_of_class(
                    training_callback, TESTING_PROCESS, test_params)

    # start training
    foreach_training(log_tool=log_tool, checkpoint=checkpoint, checkpoint_prefix=checkpoint_prefix,
                     config_loader=config_loader, train_each_round=train_each_round)


def foreach_training(log_tool, checkpoint, checkpoint_prefix, config_loader,  train_each_round):
    """
    训练管理，负责训练， 记录每轮情况
    该代码块有原先训练代码抽离
    @since 2019.11.20
    @author yuwei
    """
    # start training
    print("start training, epochs={}".format(config_loader.epoch))
    # start from last time
    print("start from epoch={}".format(log_tool.current_epoch))
    print("------------------------------------------------------------------\n")
    for epoch in range(log_tool.current_epoch, config_loader.epoch+1):
        # initial time
        start = time.time()

        # inovke each round
        train_each_round(epoch)

        # saving (checkpoint) the model every 20 epochs
        if epoch % config_loader.save_period == 0:
            # remove history checkpoints
            if config_loader.remove_history_checkpoints == True:
                train_tool.remove_history_checkpoints(
                    config_loader.checkpoints_dir)
                print("remove history checkpoints...")

            # save the checkpoint
            checkpoint.save(file_prefix=checkpoint_prefix)
            print("store current checkpoint successfully...")

        # update epoch
        log_tool.update_epoch()
        print("time taken for epoch {} is {} sec".format(
            epoch, time.time() - start))
        print("------------------------------------------------------------------\n")

    print("Total Training process finished...")


def main(config_loader):
    # load the train dataset
    train_dir = os.path.join(config_loader.data_dir, "train")
    train_loader = loader.DataLoader(train_dir, is_training=True)
    train_dataset = train_loader.load(config_loader)
    # load the test dataset
    test_dir = os.path.join(config_loader.data_dir, "test")
    test_loader = loader.DataLoader(test_dir, is_training=False)
    test_dataset = test_loader.load(config_loader)

    training_callback = None
    # judge use default callback
    callback_name = config_loader.training_callback
    if callback_name in ("default", None):
        # use default callback
        pass
    else:
        # split the module name and class name
        module_name, class_name = callback_name.split(".")
        # load training function in train_watermark
        o = importlib.import_module(module_name)
        Callback = getattr(o, class_name)
        training_callback = Callback(config_loader=config_loader)

    # train
    if config_loader.training_device in ("CPU"):
        with tf.device("/cpu:0"):
            train(train_dataset=train_dataset,
                  test_dataset=test_dataset, config_loader=config_loader, training_callback=training_callback)
    else:
        # with tf.device("/gpu:0"):
        train(train_dataset=train_dataset, test_dataset=test_dataset,
                config_loader=config_loader, training_callback=training_callback)


# bootstrap the train procedue
if __name__ == "__main__":

    # default runtime environment
    env = "./"
    # default config file
    config_file = "config.txt"

    # extern script
    extern_script = None

    # parse input arguments if exist
    if len(sys.argv) >= 2:
        options, _ = getopt.getopt(sys.argv[1:], "he:c:s:", [
                                   "help", "env=", "config=", "script="])
        print(options)
        for key, value in options:
            if key in ("-c", "--config"):
                config_file = value
            if key in ("-e", "--env"):
                env = value
            if key in ("-h", "--help"):
                print("-h -e dic -c config.txt")
            if key in ("-s", "--script"):
                extern_script = value
                

    # set current runtime environment
    os.chdir(env)
    print(os.getcwd())
    # load config file
    config_loader = config.ConfigLoader(config_file)

    if extern_script == None:
        # start training
        main(config_loader=config_loader)
    else:
        # load extern script
        module = importlib.import_module(extern_script)
        # search main function
        main_func = getattr(module, "main")
        # start
        main_func(config_loader)
