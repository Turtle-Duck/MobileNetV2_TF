import skimage
from skimage import io
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow.contrib as tc

import numpy as np
import time


class MobileNetV2(object):
    def __init__(self, input_size=224, class_num=1000):
        self.input_size = input_size
        self.is_training = tf.placeholder(tf.bool, shape=[])  # is_training
        self.normalizer = tc.layers.batch_norm
        self.bn_params = {'is_training': self.is_training, 'scale': True}
        self.class_num = class_num

        with tf.variable_scope('MobilenetV2'):
            self._create_placeholders()
            self._build_model()

    def _create_placeholders(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size, self.input_size, 3])

    def _build_model(self):
        self.i = 0
        net = tc.layers.conv2d(self.input, 32, 3, 2, normalizer_fn=self.normalizer, normalizer_params=self.bn_params,
                               activation_fn=tf.nn.relu6)
        self.output = self._inverted_bottleneck(net, 1, 16, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 24, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 24, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 320, 0)
        self.output = tc.layers.conv2d(self.output, 1280, 1, normalizer_fn=self.normalizer,
                                       normalizer_params=self.bn_params, activation_fn=tf.nn.relu6)
        with tf.variable_scope('Logits'):
            pool_kernel_size_1 = self.output.get_shape().as_list()[1]
            pool_kernel_size_2 = self.output.get_shape().as_list()[2]
            self.output = tc.layers.avg_pool2d(self.output, [pool_kernel_size_1, pool_kernel_size_2])
            self.output = tc.layers.dropout(self.output, is_training=self.is_training)
            self.output = tc.layers.conv2d(self.output, self.class_num, 1, activation_fn=None,
                                           normalizer_fn=None, scope='Conv2d_1c_1x1')
            self.output = tf.squeeze(self.output, [1, 2])

    def _inverted_bottleneck(self, input, up_sample_rate, channels, subsample):
        if self.i == 0:
            temp_scop = 'expanded_conv'
        else:
            temp_scop = 'expanded_conv_{}'.format(self.i)
        with tf.variable_scope(temp_scop):
            self.i += 1
            stride = 2 if subsample else 1
            pred_depth = input.get_shape().as_list()[-1]
            if up_sample_rate > 1:
                output = tc.layers.conv2d(input, up_sample_rate*input.get_shape().as_list()[-1], 1, scope='expand',
                                          activation_fn=tf.nn.relu6,
                                          normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            else:
                output = input
            output = tc.layers.separable_conv2d(output, None, 3, 1, stride=stride, scope='depthwise',
                                                activation_fn=tf.nn.relu6,
                                                normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            output = tc.layers.conv2d(output, channels, 1, activation_fn=tf.identity, scope='project',
                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
            if stride == 1 and pred_depth == channels:
                output = tf.add(input, output)
            return output


def normalize(img):
    img = img.astype(np.float32)
    return img / 255. - 0.5


if __name__ == '__main__':

    model_path = 'model_pretrain/mobilenet_v2_1.0_128/mobilenet_v2_1.0_128.ckpt'
    model = MobileNetV2(input_size=128, class_num=1001)
    top_k = 15
    fake_data = np.zeros(shape=(1, 128, 128, 3), dtype=np.float32)

    sess_config = tf.ConfigProto(device_count={'GPU':0})
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        variables = tf.global_variables()
        vars_restore = [var for var in variables if ("MobilenetV2" in var.name)]
        saver_restore = tf.train.Saver(vars_restore)
        saver_restore.restore(sess, model_path)

        fake_data[0] = 255.
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '1_1.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '1_2.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '2_1.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '2_2.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '3_1.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '3_2.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '3_3.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '3_4.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '3_5.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '3_6.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '3_7.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '4_1.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '4_2.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '4_3.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '4_4.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])

        img = skimage.io.imread(os.path.join('test_pic', '4_5.JPEG'))
        fake_data[0] = normalize(img)
        output = sess.run(model.output, feed_dict={model.input: fake_data, model.is_training: False})
        print(output[0].argsort()[-top_k:][::-1])


