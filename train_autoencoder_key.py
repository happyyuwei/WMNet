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

# other dependency
from model import auto
import train_tool
import config
from train import foreach_training
from loader import load_cifar

# execution
tf.enable_eager_execution()





def create_key():
    """
    create secret key, return only two value (-1,1)
    :param since 2019.11.7
    :return:
    """

    # create key
    key = np.random.rand(1, 1024)
    key = np.reshape(key, [1, 32, 32, 1])
    key[key >= 0.5] = 1
    key[key < 0.5] = -1
    # save in file
    np.save("key.npy", key)
    # to tensor
    key = tf.convert_to_tensor(key, dtype=tf.float32)
    return key


def evaluate_test_loss(test_images, image_num, encoder, decoder, key_encoder, secret_key):
    """
    evaluate the mean loss on test data
    @since 2019.9.20
    @author yuwei
    :param test_images:
    :param image_num
    :param encoder:
    :param decoder:
    :param secret_key
    :return:
    """
    # key feature
    key_output = key_encoder(secret_key, training=True)

    test_loss = 0

    for i in range(test_images.shape[0]):
        # encoder
        encoder_output = encoder(test_images[i:i + 1, :, :, :], training=True)
        # combine
        out = tf.concat([encoder_output, key_output], axis=-1)
        # decoder
        decoder_output = decoder(out, training=True)
        # calculate loss
        test_loss = test_loss + \
            auto.loss(test_images[i:i + 1, :, :, :], decoder_output)

    # calculate mean loss
    return test_loss / float(image_num)


def global_train_iterator(input_image, train_tape, key_encoder,  encoder, decoder, train_optimizer, secret_key, noise):
    """
    train iterator， 引入秘钥
    @since 2019.11.7
    @author yuwei
    :param input_image:
    :param train_tape:
    :param encoder:
    :param decoder:
    :param train_optimizer:
    :return:
    """
    # encoder
    encoder_output = encoder(input_image, training=True)

    # correct key
    if random.random() > 0.5:
        # key encoder
        key_output = key_encoder(secret_key, training=True)
        out = tf.concat([encoder_output, key_output], axis=-1)
        # out = encoder_output + key_output
        # decoder
        decoder_output = decoder(out, training=True)

        # loss
        loss = auto.loss(input_image, decoder_output)

    else:
        # if the key is incorrect, the output is totally black
        incorrent_key = np.array(secret_key).copy()
        # random bit change
        for _ in range(0, 5):
            rand_bit = int(random.random() * 1024)
            x = int(rand_bit / 32)
            y = int(rand_bit % 32)
            incorrent_key[0, x, y, 0] = -1 * incorrent_key[0, x, y, 0]

        key_output = key_encoder(incorrent_key, training=True)
        out = tf.concat([encoder_output, key_output], axis=-1)
        # decoder
        decoder_output = decoder(out, training=True)

        # loss
        loss = auto.loss(noise, decoder_output)

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

    print("load cifar data....")
    images, test_images = load_cifar(
        config_loader.data_dir, train_image_num, test_image_num)

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
    print("initial key encoder....")
    key_encoder = auto.Encoder()

    print("create noise....")
    noise = tf.zeros([1, 32, 32, 1], dtype=tf.float32)

    key_path="../../watermark/key.npy"
    secret_key = None
    if os.path.exists(key_path):
        print("load key....")
        secret_key = np.load(key_path)
        # convert to tensor
        secret_key = tf.convert_to_tensor(secret_key, dtype=tf.float32)
    else:
        print("error: key not found")
        sys.exit()

    # initial optimizer
    print("initial optimizer....")
    train_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)

    checkpoint = tf.train.Checkpoint(
        train_optimizer=train_optimizer, encoder=encoder, decoder=decoder, key_encoder=key_encoder)

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
        # image_num = 1
        # initial loss
        train_loss = 0
        # calculate image length
        image_len = images.shape[0]
        # each epoch, run all the images
        for i in range(image_len):
            # print("input_image {}".format(image_num))
            input_image = images[i:i + 1, :, :, :]
            # calculate input number
            # image_num = image_num + 1

            with tf.GradientTape() as train_tape:
                # global train
                train_loss = global_train_iterator(input_image=input_image, train_tape=train_tape, key_encoder=key_encoder,
                                                   encoder=encoder, decoder=decoder, train_optimizer=train_optimizer, secret_key=secret_key, noise=noise)

        # save test result
        if epoch % config_loader.save_period == 0:
            rand = random.randint(0, test_images.shape[0] - 1)
            # get a random image
            input_image = test_images[rand:rand + 1, :, :, :]
            # encode
            encoder_output = encoder(input_image, training=True)

            # show encoder output with secret key
            key_output = key_encoder(secret_key, training=True)

            out_positive = tf.concat([encoder_output, key_output], axis=-1)
            # decoder
            decoder_output_positive = decoder(out_positive, training=True)

            # show encoder output with wrong key
            # 变换 256， 512, 1024位数， 查看输出情况
            # 1) 变换256位, 变换左上角16*16bit
            wrong_key_256 = np.array(secret_key)
            wrong_key_256[0, 0:16, 0:16, 0] = - \
                1*wrong_key_256[0, 0:16, 0:16, 0]
            wrong_key_256 = tf.convert_to_tensor(wrong_key_256)
            wrong_key_output_256 = key_encoder(wrong_key_256, training=True)
            out_negitive_256 = tf.concat(
                [encoder_output, wrong_key_output_256], axis=-1)
            # decoder
            decoder_output_negitive_256 = decoder(
                out_negitive_256, training=True)

            # 2) 变换512位, 变换上方32*16bit
            wrong_key_512 = np.array(secret_key)
            wrong_key_512[0, 0:32, 0:16, 0] = - \
                1*wrong_key_512[0, 0:32, 0:16, 0]
            wrong_key_512 = tf.convert_to_tensor(wrong_key_512)
            wrong_key_output_512 = key_encoder(wrong_key_512, training=True)
            out_negitive_512 = tf.concat(
                [encoder_output, wrong_key_output_512], axis=-1)
            # decoder
            decoder_output_negitive_512 = decoder(
                out_negitive_512, training=True)

            # 3) 变换1024位, 全部变换
            wrong_key_1024 = -1*secret_key
            wrong_key_output_1024 = key_encoder(wrong_key_1024, training=True)
            out_negitive_1024 = tf.concat(
                [encoder_output, wrong_key_output_1024], axis=-1)
            # decoder
            decoder_output_negitive_1024 = decoder(
                out_negitive_1024, training=True)

            titles = ["IN", "EN", "DE+", "DE-256", "DE-512", "DE-1024"]
            image_list = [input_image, tf.reshape(encoder_output, [1, 128, 128, 3]), decoder_output_positive,
                          decoder_output_negitive_256, decoder_output_negitive_512, decoder_output_negitive_1024]

            # record
            log_tool.save_image_list(image_list=image_list, title_list=titles)
            log_tool.save_image_list(
                image_list=image_list, title_list=titles)
            # evaluate in test data
            test_loss = evaluate_test_loss(test_images=test_images, image_num=test_image_num,
                                           encoder=encoder, key_encoder=key_encoder, decoder=decoder, secret_key=secret_key)
            # save loss and test loss in log file
            log_tool.save_loss(train_loss=train_loss, test_loss=test_loss)

    # start training
    foreach_training(log_tool=log_tool, checkpoint=checkpoint, checkpoint_prefix=checkpoint_prefix,
                     config_loader=config_loader, train_each_round=train_each_round)


def main(config_loader):
    print("This is the script for training auto encoder-decoder with secret key.")
    # train
    if config_loader.training_device in ("CPU"):
        with tf.device("/cpu:0"):
            train(config_loader=config_loader)
    else:
        with tf.device("/gpu:0"):
            train(config_loader=config_loader)
