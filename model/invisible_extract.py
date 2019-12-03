import tensorflow as tf
from model.cgan import Downsample
from model.cgan import Upsample

"""
first version start
@since 2019.9.14
@author yuwei

second version start, two convolution block instead of one block
@since 2019.9.22
@author yuwei
"""


class Conv(tf.keras.Model):
    def __init__(self, filters, size, apply_batchnorm=True, relu=True):
        super(Conv, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.relu = relu
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=True)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        if self.relu is True:
            x = tf.nn.relu(x)
        return x


class ExtractInvisible(tf.keras.Model):

    def __init__(self):
        super(ExtractInvisible, self).__init__()
        # extend channel
        self.first_conv = Conv(32, 1, apply_batchnorm=True)

        # first block comes from here
        # channel 1 conv
        self.block1_channel1_conv1 = Conv(32, 1, apply_batchnorm=True)
        # channel 2 conv
        self.block1_channel2_conv1 = Conv(32, 1, apply_batchnorm=True)
        self.block1_channel2_conv2 = Conv(32, 3, apply_batchnorm=True)
        # channel 3 conv
        self.block1_channel3_conv1 = Conv(32, 1, apply_batchnorm=True)
        self.block1_channel3_conv2 = Conv(32, 3, apply_batchnorm=True)
        self.block1_channel3_conv3 = Conv(32, 3, apply_batchnorm=True)
        # channel combine1 conv
        self.block1_combine_conv = Conv(32, 1, apply_batchnorm=True, relu=False)

        # second block comes from here
        # channel 1 conv
        self.block2_channel1_conv1 = Conv(32, 1, apply_batchnorm=True)
        # channel 2 conv
        self.block2_channel2_conv1 = Conv(32, 1, apply_batchnorm=True)
        self.block2_channel2_conv2 = Conv(32, 3, apply_batchnorm=True)
        # channel 3 conv
        self.block2_channel3_conv1 = Conv(32, 1, apply_batchnorm=True)
        self.block2_channel3_conv2 = Conv(32, 3, apply_batchnorm=True)
        self.block2_channel3_conv3 = Conv(32, 3, apply_batchnorm=True)
        # channel combine1 conv
        self.block2_combine_conv = Conv(32, 1, apply_batchnorm=True, relu=False)

        # final convolution comes here
        self.final_conv = Conv(3, 1, apply_batchnorm=True, relu=False)

    @tf.contrib.eager.defun
    def call(self, x, training):
        # extend channel
        x = self.first_conv(x, training)
        # block 1 comes from here
        # channel 1
        block1_channel1_x1 = self.block1_channel1_conv1(x, training=training)
        # channel 2
        block1_channel2_x1 = self.block1_channel2_conv1(x, training=training)
        block1_channel2_x2 = self.block1_channel2_conv2(block1_channel2_x1, training=training)
        # channel 2
        block1_channel3_x1 = self.block1_channel3_conv1(x, training=training)
        block1_channel3_x2 = self.block1_channel3_conv2(block1_channel3_x1, training=training)
        block1_channel3_x3 = self.block1_channel3_conv3(block1_channel3_x2, training=training)
        # combine
        block1_combine_x1 = tf.concat([block1_channel1_x1, block1_channel2_x2, block1_channel3_x3], axis=-1)
        block1_combine_x2 = self.block1_combine_conv(block1_combine_x1, training=training)
        # add
        block1_out = x + block1_combine_x2
        # relu after add residual
        block1_out = tf.nn.relu(block1_out)

        # block 2 comes from here
        # channel 1
        block2_channel1_x1 = self.block2_channel1_conv1(block1_out, training=training)
        # channel 2
        block2_channel2_x1 = self.block2_channel2_conv1(block1_out, training=training)
        block2_channel2_x2 = self.block2_channel2_conv2(block2_channel2_x1, training=training)
        # channel 2
        block2_channel3_x1 = self.block2_channel3_conv1(block1_out, training=training)
        block2_channel3_x2 = self.block2_channel3_conv2(block2_channel3_x1, training=training)
        block2_channel3_x3 = self.block2_channel3_conv3(block2_channel3_x2, training=training)
        # combine
        block2_combine_x1 = tf.concat([block2_channel1_x1, block2_channel2_x2, block2_channel3_x3], axis=-1)
        block2_combine_x2 = self.block2_combine_conv(block2_combine_x1, training=training)
        # add
        block2_out = block1_out + block2_combine_x2
        block2_out = tf.nn.relu(block2_out)

        # final convolution comes from here
        out = self.final_conv(block2_out, training=training)

        # use tanh instead, because sigmod is [0,1] and tanh is [-1,1]
        # @since 2019.9.21
        # author yuwei
        out = tf.nn.tanh(out)
        return out

# class ExtractInvisible(tf.keras.Model):
#
#     def __init__(self):
#         super(ExtractInvisible, self).__init__()
#
#         # conv block1
#         self.conv_block1 = ConvBlock()
#         #conv block2
#         self.conv_block2 = ConvBlock()
#
#     @tf.contrib.eager.defun
#     def call(self, x, training):
#         x1 = self.conv_block1(x, training=training)
#         x2 = self.conv_block2(x1, training=training)
#         return x2
