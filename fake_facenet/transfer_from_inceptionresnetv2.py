#!/usr/bin/env python
# coding: utf-8

import copy
import pathlib
import random

import tensorflow as tf
# from tensorflow.keras.layers import Input, AveragePooling2D, concatenate, Dense, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow._api.v1.keras.layers import Input, AveragePooling2D, concatenate, Dense, Activation, Flatten

tf.enable_eager_execution()

# State dataset directory where the tfrecord files are located
dataset_dir = '/Users/hchan/Downloads/CASIA-FaceV5-Crop'

# State where your log file is at. If it doesn't exist, create it.
log_dir = './log'

# State where your checkpoint file is
checkpoint_file = './inception_resnet_v2_2016_08_30.ckpt'

batch_size = 32


def create_dataset():
    for pt1, pt2 in zip(path_sf1_str, path_sf2_str):
        img_tensor1, name1 = get_img_and_name(pt1)
        img_tensor2, name2 = get_img_and_name(pt2)

        label = 1 if name1 == name2 else 0
        yield {'input1': img_tensor1, 'input2': img_tensor2}, label
    for cursor in range(0, len(piclist_str) - 5, 5):
        for _ in range(5):
            rand1 = random.randint(0, 4)
            rand2 = random.randint(0, 4)
            pt1 = piclist_str[cursor + rand1]
            pt2 = piclist_str[cursor + rand2]
            img_tensor1, name1 = get_img_and_name(pt1)
            img_tensor2, name2 = get_img_and_name(pt2)

            label = 1 if name1 == name2 else 0
            yield {'input1': img_tensor1, 'input2': img_tensor2}, label


def get_img_and_name(img_path):
    img_raw = tf.io.read_file(img_path)
    img_tensor = tf.image.decode_bmp(img_raw, channels=3)
    img_tensor = tf.image.resize(img_tensor, [299, 299])
    img_tensor = img_tensor / 127.5 - 1
    name = img_path.split('/')[5]
    return img_tensor, name


# for x, y in image_label_ds:
#     model.fit([x, x], y)


def generate_inceptionResnetv2_based_model():
    irv2 = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False)
    irv2.trainable = False
    # This returns a tensor
    input1 = Input(shape=(299, 299, 3), name='input1')
    input2 = Input(shape=(299, 299, 3), name='input2')
    out1 = irv2(input1)
    out2 = irv2(input2)
    averPool = AveragePooling2D(pool_size=(8, 8))
    out1 = averPool(out1)
    out2 = averPool(out2)
    y = concatenate([out1, out2])
    dense = Dense(1)
    y = dense(y)
    activation = Activation('tanh')
    y = activation(y)
    y = Flatten()(y)
    model = Model(inputs=[input1, input2], outputs=y)
    return model


if __name__ == '__main__':
    data_root = pathlib.Path(dataset_dir)
    piclist = list(data_root.glob('*/*.bmp'))
    total_data_size = len(piclist)
    batch_size = 32
    steps_per_epoch = tf.math.ceil(total_data_size / batch_size)
    print("Total image size:", total_data_size)

    piclist_str = list(map(str, piclist))

    path_sf1 = copy.deepcopy(piclist)
    path_sf2 = copy.deepcopy(piclist)
    random.shuffle(path_sf1)
    random.shuffle(path_sf2)

    path_sf1_str = list(map(str, path_sf1))
    path_sf2_str = list(map(str, path_sf2))

    # 生成图片label

    label_list = [int(pic_path.parent.name) for pic_path in piclist]

    # 构建Dataset
    image_label_ds = tf.data.Dataset.from_generator(create_dataset,
                                                    ({'input1': tf.int64, 'input2': tf.int64}, tf.int64))

    image_label_ds = image_label_ds.repeat()
    image_label_ds = image_label_ds.shuffle(buffer_size=64)
    steps_per_epoch = tf.math.ceil(4900 / batch_size)
    image_label_ds = image_label_ds.batch(batch_size)

    model = generate_inceptionResnetv2_based_model()

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    model.fit_generator(image_label_ds, epochs=5, steps_per_epoch=steps_per_epoch)
