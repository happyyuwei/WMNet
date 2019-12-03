"""
In this area, we just train visible watermarks
@since 2019.9.12
"""
# tensorflow deepling framework
import tensorflow as tf

# third part dependency
import numpy as np
# normal dependency
import os
import time
import sys
import getopt
import random
import pickle

# other dependency
from model import auto
import train_tool
import config
import loader
from train import foreach_training

# execution
tf.enable_eager_execution()

def create_mask_list():
    """
    create mask list, currently, there are four kind,left,right,up,and down crop, each crop area is 20px width
    @since 2019.9.21
    @author yuwei
    :return:
    """
    left_crop = np.ones(shape=[1, 128, 128, 3], dtype=np.float32)
    left_crop[:, :, 0:20, :] = 0

    right_crop = np.ones(shape=[1, 128, 128, 3], dtype=np.float32)
    # the last 20px, in python, we use -20
    right_crop[:, :, -20:, :] = 0

    up_crop = np.ones(shape=[1, 128, 128, 3], dtype=np.float32)
    up_crop[:, 0:20, :, :] = 0

    down_crop = np.ones(shape=[1, 128, 128, 3], dtype=np.float32)
    down_crop[:, -20:, :, :] = 0

    # add in the list
    mask_list = [left_crop, right_crop, up_crop, down_crop]
    for i in range(len(mask_list)):
        mask_list[i] = tf.convert_to_tensor(mask_list[i])
        # change size to meet the encoder output
        mask_list[i] = tf.reshape(mask_list[i], [1, 32, 32, 48])
    return mask_list


def crop_image(encoder_output, mask_list):
    """
    to train the network more robust, there are some probability to crop the output of the encoder
    the mask is 1 not crop and 0 crop.
    the encoder_output is -1 which is cropped.
    :param encoder_output:
    :param mask_list
    :return:
    """
    # there are 50% to crop the image
    if random.random() < 0.5:
        # choose a mask
        rand = random.randint(0, len(mask_list) - 1)
        mask = mask_list[rand]
        encoder_output = tf.multiply(encoder_output, mask) + mask - 1

    return encoder_output


def evaluate_test_loss(test_images, image_num, encoder, decoder):
    """
    evaluate the mean loss on test data
    @since 2019.9.20
    @author yuwei
    :param test_images:
    :param image_num
    :param encoder:
    :param decoder:
    :return:
    """
    test_loss = 0
    for i in range(test_images.shape[0]):
        # encoder
        encoder_output = encoder(test_images[i:i + 1, :, :, :], training=True)
        # decoder
        decoder_output = decoder(encoder_output, training=True)
        # calculate loss
        test_loss = test_loss + \
            auto.loss(test_images[i:i + 1, :, :, :], decoder_output)

    # calculate mean loss
    return test_loss / float(image_num)


def global_train_iterator(input_image, mask_list, train_tape, encoder, decoder, train_optimizer):
    """
    train iterator
    @since 2019.9.18
    @author yuwei
    :param input_image:
    :param mask_list
    :param train_tape:
    :param encoder:
    :param decoder:
    :param train_optimizer:
    :return:
    """
    # encoder
    encoder_output = encoder(input_image, training=True)

    # crop in random, to train robust decode
    encoder_output = crop_image(encoder_output, mask_list)

    # decoder
    decoder_output = decoder(encoder_output, training=True)

    # loss
    loss = auto.loss(input_image, decoder_output)

    # gather variables
    # Todo i am not sure it is right to gather like this way in Tensorflow 2.0
    variables = encoder.variables + decoder.variables

    # calculate gradients and optimize
    gradients = train_tape.gradient(target=loss, sources=variables)
    train_optimizer.apply_gradients(grads_and_vars=zip(gradients, variables))

    return loss


def train(config_loader):
    """
    train main loop
    :param config_loader:
    :return:
    """
    # # since the number of the ship images in cifar10-batch1 is 1025,
    # the first 1000 images are for training, then the next 25 images are for testing
    train_image_num = 1000
    test_image_num = 25

    # start
    print("initial training progress....")

    # load data
    if "cifar" in config_loader.data_dir:
        print("load cifar data....")
        images, test_images = loader.load_cifar(
            config_loader.data_dir, train_image_num, test_image_num)
    elif "mnist" in config_loader.data_dir:
        print("load mnist data....")
        images, test_images = loader.load_mnist(
            config_loader.data_dir, train_image_num, test_image_num)
    else:
        print("neither cifar nor mnist data found...")
        return

    print("create mask list....")
    mask_list = create_mask_list()

    # checkpoint instance
    print("initial checkpoint....")
    checkpoint_prefix = os.path.join(config_loader.checkpoints_dir, "ckpt")

    # The call function of Generator and Discriminator have been decorated
    # with tf.contrib.eager.defun()
    # We get a performance speedup if defun is used (~25 seconds per epoch)
    print("initial encoder....")
    encoder = auto.Encoder()
    print("initial decoder....")
    decoder = auto.Decoder()

    # initial optimizer
    print("initial optimizer....")
    train_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)

    checkpoint = tf.train.Checkpoint(
        train_optimizer=train_optimizer, encoder=encoder, decoder=decoder)

    print("initial train log....")
    log_tool = train_tool.LogTool(
        config_loader.log_dir, config_loader.save_period)

    # restoring the latest checkpoint in checkpoint_dir if necessary
    if config_loader.load_latest_checkpoint is True:
        checkpoint.restore(tf.train.latest_checkpoint(
            config_loader.checkpoints_dir))
        print("load latest checkpoint....")

    
    def train_each_round(epoch):
        # initial input number
        image_num = 1
        # initial loss
        train_loss = 0
        # calculate image length
        image_len = images.shape[0]
        # each epoch, run all the images
        for i in range(image_len):
            # print("input_image {}".format(image_num))
            input_image = images[i:i + 1, :, :, :]
            # calculate input number
            image_num = image_num + 1

            with tf.GradientTape() as train_tape:
                # global train
                train_loss = global_train_iterator(input_image=input_image, mask_list=mask_list, train_tape=train_tape,
                                                   encoder=encoder, decoder=decoder, train_optimizer=train_optimizer)

        # save test result
        if epoch % config_loader.save_period == 0:
            rand = random.randint(0, test_images.shape[0] - 1)
            # get a random image
            input_image = test_images[rand:rand + 1, :, :, :]
            # show encoder output
            encoder_output = encoder(input_image, training=True)
            # crop the encoder output
            encoder_output = crop_image(encoder_output, mask_list)
            # decoder
            decoder_output = decoder(encoder_output, training=True)
            titles = ["IN", "EN", "DE"]
            image_list = [input_image, tf.reshape(
                encoder_output, [1, 128, 128, 3]), decoder_output]
            log_tool.save_image_list(image_list=image_list, title_list=titles)
            # evaluate in test data
            test_loss = evaluate_test_loss(test_images=test_images, image_num=test_image_num, encoder=encoder,
                                           decoder=decoder)
            # save loss and test loss in log file
            log_tool.save_loss(train_loss=train_loss, test_loss=test_loss)

     # start training
    foreach_training(log_tool=log_tool, checkpoint=checkpoint, checkpoint_prefix=checkpoint_prefix,
                     config_loader=config_loader, train_each_round=train_each_round)


def main(config_loader):
    # train
    if config_loader.training_device in ("CPU"):
        with tf.device("/cpu:0"):
            train(config_loader=config_loader)
    else:
        with tf.device("/gpu:0"):
            train(config_loader=config_loader)


