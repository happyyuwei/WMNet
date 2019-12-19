import tensorflow as tf
import numpy as np
import os
import pickle


def load_image(image_file, input_num, is_train, image_width, image_height, flap_probability, crop_width, crop_height):
    """
    loader images, the left is grand truth, the right list is input,
    currently, one or two inputs are supported,
    use input_num to point out.
    :param image_file:
    :param input_num:
    :param is_train:
    :param image_width:
    :param image_height:
    :param flap_probability
    :param input_concat
    :param crop_width
    :param crop_height
    :return:
    """
    image = tf.read_file(image_file)
    # image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    # only three channel currently
    # author yuwei
    # @since 2019.9.14
    image = image[:, :, 0:3]

    # get image width
    w = tf.shape(image)[1]

    w = w // (input_num + 1)
    # the left is label, the right is input, 1 and 2
    real_image = image[:, :w, :]
    input_image_1 = image[:, w:2 * w, :]
    if input_num == 2:
        input_image_2 = image[:, 2 * w:3 * w, :]
    else:
        # useless input 2
        input_image_2 = input_image_1

    # to float 32
    input_image_1 = tf.cast(input_image_1, tf.float32)
    input_image_2 = tf.cast(input_image_2, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    if is_train:
        # random
        # resizing to 286 x 286 x 3
        # we have to inform that the image_height is in front of  image_width,
        # because image_height is the row number of a matrix and the image_width is the column number of it.
        # The size of a matrix can be described by [row, column], so the image matrix is [image_height, image_width]
        input_image_1 = tf.image.resize_images(input_image_1, [image_height + crop_height, image_width + crop_width],
                                               align_corners=True,
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        input_image_2 = tf.image.resize_images(input_image_2, [image_height + crop_height, image_width + crop_width],
                                               align_corners=True,
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize_images(real_image, [image_height + crop_height, image_width + crop_width],
                                            align_corners=True,
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if crop_width != 0 or crop_height != 0:
            # randomly cropping to 256 x 256 x 3
            stacked_image = tf.stack(
                [input_image_1, input_image_2, real_image], axis=0)
            cropped_image = tf.random_crop(
                stacked_image, size=[3, image_height, image_width, 3])
            input_image_1, input_image_2, real_image = cropped_image[
                0], cropped_image[1], cropped_image[2]

        # remove flip, i just think it is useless in this situation, but i don't know
        # maybe useful in some standard datasets like facade
        # @since 2018.1.18
        if np.random.random() < flap_probability:
            # random mirroring
            input_image_1 = tf.image.flip_left_right(input_image_1)
            input_image_2 = tf.image.flip_left_right(input_image_2)
            real_image = tf.image.flip_left_right(real_image)

    else:
        input_image_1 = tf.image.resize_images(input_image_1, size=[image_height, image_width],
                                               align_corners=True, method=2)
        input_image_2 = tf.image.resize_images(input_image_2, size=[image_height, image_width],
                                               align_corners=True, method=2)
        real_image = tf.image.resize_images(real_image, size=[image_height, image_width],
                                            align_corners=True, method=2)

    # normalizing the images to [-1, 1]
    input_image_1 = (input_image_1 / 127.5) - 1
    if input_num == 2:
        input_image_2 = (input_image_2 / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    # concat if necessary
    if input_num == 2:
        input_image = tf.concat([input_image_1, input_image_2], axis=-1)
    else:
        input_image = input_image_1

    return input_image, real_image


def load_image_new(image_file, input_num, is_train, image_width, image_height, flap_probability, crop_width, crop_height):

    image = tf.read_file(image_file)
    # image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    # only three channel currently
    # author yuwei
    # @since 2019.9.14
    image = image[:, :, 0:3]

    # get image width
    w = tf.shape(image)[1]

    w = w // (input_num + 1)
    # the left is label, the right is input, 1 and 2
    real_image = image[:, :w, :]
    input_image_1 = image[:, w:2 * w, :]
    if input_num == 2:
        input_image_2 = image[:, 2 * w:3 * w, :]
    else:
        # useless input 2
        input_image_2 = input_image_1

    # to float 32
    input_image_1 = tf.cast(input_image_1, tf.float32)
    input_image_2 = tf.cast(input_image_2, tf.float32)
    real_image = tf.cast(real_image, tf.float32)


    # normalizing the images to [-1, 1]
    input_image_1 = (input_image_1 / 127.5) - 1
    if input_num == 2:
        input_image_2 = (input_image_2 / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    # concat if necessary
    if input_num == 2:
        input_image = tf.concat([input_image_1, input_image_2], axis=-1)
    else:
        input_image = input_image_1

    return input_image, real_image



class DataLoader:

    def __init__(self, data_dir, is_training):
        """

        :param train_dir:
        :param is_training:
        """
        self.data_dir = data_dir
        self.is_training = is_training

    def load(self, config_loader):
        """

        :param config_loader:
        :return:
        """
        # get dataset file list
        dataset = tf.data.Dataset.list_files(
            os.path.join(self.data_dir, "*.jpg"))
        # recalculate buffer size
        # if buffer_size<=0， adapted the buffer size
        # then buffer size is the dataset buffer size
        # recommend buffer_size=0 to adapt the dataset size
        # @since 2019.1.18
        # @author yuwei
        # @version 0.93
        if config_loader.buffer_size <= 0:
            config_loader.buffer_size = len(os.listdir(self.data_dir))

        # shuffle the list if necessary
        if self.is_training is True:
            dataset = dataset.shuffle(buffer_size=config_loader.buffer_size)

        # pretreat images
        dataset = dataset.map(
            lambda x: load_image_new(x, config_loader.input_num, self.is_training, config_loader.image_width,
                                 config_loader.image_height, config_loader.data_flip, config_loader.crop_width,
                                 config_loader.crop_height))

        # return batch
        return dataset.batch(config_loader.batch_size)


def load_cifar(cifar_file, train_num, test_num):
    """
    this function is to use to load cifar images to tensor
    @update 2019.11.30
    移至loader文件作为基础设施
    @since 2019.9.20
    @author yuwei
    :param cifar_file:
    :param train_num:
    :param test_num:
    :return:
    """
    with open(cifar_file, 'rb') as fo:
        # load dictionary
        dict = pickle.load(fo, encoding='bytes')
        # load labels
        labels = dict[b'labels']
        # load images binary
        imgs = dict[b'data']

    # image list
    image_list = []
    # currently, I only use ship images, the label is 8
    for i in range(len(labels)):
        if labels[i] == 8:
            # reshape the image to RGB channel
            img = np.reshape(imgs[i], (3, 32, 32))
            temp = np.zeros([32, 32, 3])
            temp[:, :, 0] = img[0, :, :]
            temp[:, :, 1] = img[1, :, :]
            temp[:, :, 2] = img[2, :, :]
            img = temp
            image_list.append(img)

    image_array = np.zeros([len(image_list), 32, 32, 3])
    # to image array, the first axis is image number, then the three change is RGB
    for i in range(len(image_array)):
        image_array[i, :, :, :] = image_list[i]

    # to float32 tensor
    image_array = tf.convert_to_tensor(image_array.astype("float32"))
    image_array = (image_array - 127.5) / 127.5

    return image_array[0:train_num, :, :, :], image_array[train_num:train_num + test_num, :, :, :]


def load_mnist(mnist_file, train_num, test_num):
    """
    @update 2019.12.3
    移至loader文件作为基础设施
    load the mnist dataset and change it to binary images
    :param mnist_file:
    :param train_num:
    :param test_num:
    :return:
    """
    # load file，the file [0,255]
    images = np.load(mnist_file)
    # change shape
    images = images.reshape(images.shape[0], 28, 28, 1).astype("float32")
    images = tf.image.resize_images(images, [32, 32], method=0)
    images = (images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    images = np.array(images)
    # binary value -1 and 1
    images[images >= 0] = 1
    images[images < 0] = -1
    # convert to tensor
    images = tf.convert_to_tensor(images.astype("float32"))
    # the first is train images and the second is test images
    return images[0:train_num, :, :, :], images[train_num:train_num + test_num, :, :, :]
