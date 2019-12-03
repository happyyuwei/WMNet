import tensorflow as tf

"""
This model is to train an auto encoder decoder, 
which will encode the watermark to a larger bitstream and decode to itself.
Currenly, the input size is 32*32*3
The first version is stable.
@since 2019.9.20
@version author
"""


class Conv(tf.keras.Model):
    def __init__(self, filters, size, apply_batchnorm=True):
        super(Conv, self).__init__()
        self.apply_batchnorm = apply_batchnorm
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
        return x


class Encoder(tf.keras.Model):

    def __init__(self):
        super(Encoder, self).__init__()

        # self.encoder1 = Conv(12, 4, apply_batchnorm=True)
        # self.encoder2 = Conv(12, 1, apply_batchnorm=True)
        self.encoder1 = Conv(24, 3, apply_batchnorm=True)
        self.encoder2 = Conv(48, 3, apply_batchnorm=True)

    @tf.contrib.eager.defun
    def call(self, x, training):
        x1 = self.encoder1(x, training=training)
        x1 = tf.nn.relu(x1)
        x2 = self.encoder2(x1, training=training)
        # fix bug. Tanh is used instead of sigmoid because sigmoid is [0,1] but the output is [-1,1]
        # @since 2019.9.20
        # @author yuwei
        out = tf.nn.tanh(x2)
        return out


class Decoder(tf.keras.Model):

    def __init__(self, output_channel=3):
        """
        the default output is 3, currently, we introduce output channel, because we will also try 32*32*1 in MNIST
        @since 2019.9.21
        @author yuwei
        :param output_channel:
        """
        super(Decoder, self).__init__()

        # self.decoder1 = Conv(12, 1, apply_batchnorm=True)
        # self.decoder2 = Conv(1, 4, apply_batchnorm=True)
        self.decoder1 = Conv(24, 3, apply_batchnorm=True)
        self.decoder2 = Conv(output_channel, 3, apply_batchnorm=True)

    @tf.contrib.eager.defun
    def call(self, x, training):
        x1 = self.decoder1(x, training=training)
        x1 = tf.nn.relu(x1)
        x2 = self.decoder2(x1, training=training)
        out = tf.nn.tanh(x2)
        return out


def loss(input_image, outut):
    """
    unsupervise learning, the loss the mse between the output and itself
    :param input:
    :param outut:
    :return:
    """
    return tf.reduce_mean(tf.square(input_image - outut))
