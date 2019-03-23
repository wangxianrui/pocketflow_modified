import tensorflow as tf


class PeleeNet:
    def __init__(self):
        self.inner_channels = [16, 32, 64, 64]
        self.extra_channel = 16

    def conv_bn_relu(self, inputs, is_training, output_channel, kernel_size, stride, padding='same', use_relu=True):
        conv = tf.layers.Conv2D(output_channel, kernel_size, stride, padding, use_bias=False)(inputs)
        conv_bn = tf.layers.BatchNormalization()(conv, training=is_training)
        if use_relu:
            return tf.nn.relu(conv_bn)
        else:
            return conv_bn

    def stem_block(self, inputs, is_training):
        stem1 = self.conv_bn_relu(inputs, is_training, 32, 3, 2)
        stem2 = self.conv_bn_relu(stem1, is_training, 16, 1, 1)
        stem3 = self.conv_bn_relu(stem2, is_training, 32, 3, 2)
        stem1_pool = tf.layers.MaxPooling2D(2, 2)(stem1)
        stem_cat = tf.concat([stem3, stem1_pool], axis=-1)
        stem_final = self.conv_bn_relu(stem_cat, is_training, 32, 1, 1)
        return stem_final

    def dense_block(self, inputs, is_training, inner_channel, extra_channel):
        dense_a1 = self.conv_bn_relu(inputs, is_training, inner_channel, 1, 1)
        dense_a2 = self.conv_bn_relu(dense_a1, is_training, extra_channel, 3, 1)
        dense_b1 = self.conv_bn_relu(inputs, is_training, inner_channel, 1, 1)
        dense_b2 = self.conv_bn_relu(dense_b1, is_training, extra_channel, 3, 1)
        dense_b3 = self.conv_bn_relu(dense_b2, is_training, extra_channel, 3, 1)
        dense_final = tf.concat([inputs, dense_a2, dense_b3], axis=-1)
        return dense_final

    def transition_block(self, inputs, is_training, is_pooling):
        input_channel = inputs.get_shape().as_list()[3]
        if is_pooling:
            trans_0 = self.conv_bn_relu(inputs, is_training, input_channel, 1, 1)
            trans_pool = tf.layers.MaxPooling2D(2, 2)(trans_0)
            return trans_pool
        else:
            trans_0 = self.conv_bn_relu(inputs, is_training, input_channel, 1, 1)
            return trans_0

    def __call__(self, inputs, is_training):

        with tf.variable_scope('stage_0'):
            with tf.variable_scope('stem_block'):
                stage_0 = self.stem_block(inputs, is_training)

        with tf.variable_scope('stage_1'):
            stage_1_input = stage_0
            scope_name = 'dense_block_%d'
            for i in range(3):
                with tf.variable_scope(scope_name % i):
                    stage_1_input = self.dense_block(stage_1_input, is_training, self.inner_channels[0],
                                                     self.extra_channel)
            with tf.variable_scope('transition_block'):
                stage_1 = self.transition_block(stage_1_input, is_training, is_pooling=True)

        with tf.variable_scope('stage_2'):
            stage_2_input = stage_1
            scope_name = 'dense_block_%d'
            for i in range(4):
                with tf.variable_scope(scope_name % i):
                    stage_2_input = self.dense_block(stage_2_input, is_training, self.inner_channels[1],
                                                     self.extra_channel)
            with tf.variable_scope('transition_block'):
                stage_2 = self.transition_block(stage_2_input, is_training, is_pooling=True)

        with tf.variable_scope('stage_3'):
            stage_3_input = stage_2
            scope_name = 'dense_block_%d'
            for i in range(8):
                with tf.variable_scope(scope_name % i):
                    stage_3_input = self.dense_block(stage_3_input, is_training, self.inner_channels[2],
                                                     self.extra_channel)
            with tf.variable_scope('transition_block'):
                stage_3 = self.transition_block(stage_3_input, is_training, is_pooling=True)

        with tf.variable_scope('stage_4'):
            stage_4_input = stage_3
            scope_name = 'dense_block_%d'
            for i in range(6):
                with tf.variable_scope(scope_name % i):
                    stage_4_input = self.dense_block(stage_4_input, is_training, self.inner_channels[3],
                                                     self.extra_channel)
            with tf.variable_scope('transition_block'):
                stage_4 = self.transition_block(stage_4_input, is_training, is_pooling=False)

        return stage_3_input, stage_4


class PeleeNetClassify:
    def __init__(self, num_classes):
        self.backbone = PeleeNet()
        self.num_classes = num_classes

    def __call__(self, inputs, is_training):
        _, feature = self.backbone(inputs, is_training)
        pooling = tf.layers.AveragePooling2D(7, 1)(feature)
        flatten = tf.layers.Flatten()(pooling)
        dense = tf.layers.Dense(self.num_classes)(flatten)
        return dense


class PeleeNetSSD:
    def __init__(self, num_classes, anchor_depth_per_layer):
        self.peleenet = PeleeNet()
        self.extra_output_channel = 256
        self.num_classes = num_classes
        self.anchor_depth_per_layer = anchor_depth_per_layer

    def conv_bn_relu(self, inputs, is_training, output_channel, kernel_size, stride, padding='same', use_relu=True):
        conv = tf.layers.Conv2D(output_channel, kernel_size, stride, padding, use_bias=False)(inputs)
        conv_bn = tf.layers.BatchNormalization()(conv, training=is_training)
        if use_relu:
            return tf.nn.relu(conv_bn)
        else:
            return conv_bn

    def add_extra(self, inputs, is_training, output_channel):
        a2 = self.conv_bn_relu(inputs, is_training, output_channel, 1, 1, use_relu=False)
        b2a = self.conv_bn_relu(inputs, is_training, int(output_channel / 2), 1, 1)
        b2b = self.conv_bn_relu(b2a, is_training, int(output_channel / 2), 3, 1)
        b2c = self.conv_bn_relu(b2b, is_training, output_channel, 1, 1, use_relu=False)
        return a2 + b2c

    def multibox_layer(self, feature_layers):
        locations = []
        classes = []
        for i, feature in enumerate(feature_layers):
            locations.append(
                tf.layers.Conv2D(self.anchor_depth_per_layer[i] * 4, kernel_size=3, padding='same',
                                 use_bias=True)(feature))
            classes.append(
                tf.layers.Conv2D(self.anchor_depth_per_layer[i] * self.num_classes, kernel_size=3,
                                 padding='same', use_bias=True)(feature))
        return locations, classes

    def __call__(self, inputs, is_training):
        feature_layers = []

        # backbone
        stage3, stage4 = self.peleenet(inputs, is_training)

        with tf.variable_scope('extra_pm2'):
            pm2_inputs = stage3
            pm2_res = self.add_extra(pm2_inputs, is_training, self.extra_output_channel)
            feature_layers.append(pm2_res)
            feature_layers.append(pm2_res)

        with tf.variable_scope('extra_pm3'):
            pm3_inputs = stage4
            pm3_res = self.add_extra(pm3_inputs, is_training, self.extra_output_channel)
            feature_layers.append(pm3_res)

        with tf.variable_scope('extra_pm3_to_pm4'):
            pm3_to_pm4 = tf.layers.Conv2D(self.extra_output_channel, 1, 1, activation=tf.nn.relu)(pm3_inputs)
            pm3_to_pm4 = tf.layers.Conv2D(self.extra_output_channel, 3, 2, padding='same',
                                          activation=tf.nn.relu)(pm3_to_pm4)

        with tf.variable_scope('extra_pm4'):
            pm4_inputs = pm3_to_pm4
            pm4_res = self.add_extra(pm4_inputs, is_training, self.extra_output_channel)
            feature_layers.append(pm4_res)

        with tf.variable_scope('extra_pm4_to_pm5'):
            pm4_to_pm5 = tf.layers.Conv2D(self.extra_output_channel, 1, 1, activation=tf.nn.relu)(pm4_inputs)
            pm4_to_pm5 = tf.layers.Conv2D(self.extra_output_channel, 3, 1, activation=tf.nn.relu)(pm4_to_pm5)

        with tf.variable_scope('extra_pm5'):
            pm5_inputs = pm4_to_pm5
            pm5_res = self.add_extra(pm5_inputs, is_training, self.extra_output_channel)
            feature_layers.append(pm5_res)

        with tf.variable_scope('extra_pm5_to_pm6'):
            pm5_to_pm6 = tf.layers.Conv2D(self.extra_output_channel, 1, 1, activation=tf.nn.relu)(pm5_inputs)
            pm5_to_pm6 = tf.layers.Conv2D(self.extra_output_channel, 3, 1, activation=tf.nn.relu)(pm5_to_pm6)

        with tf.variable_scope('extra_pm6'):
            pm6_inputs = pm5_to_pm6
            pm6_res = self.add_extra(pm6_inputs, is_training, self.extra_output_channel)
            feature_layers.append(pm6_res)

        with tf.variable_scope('mutibox_layer'):
            locations, classes = self.multibox_layer(feature_layers)
        return locations, classes
