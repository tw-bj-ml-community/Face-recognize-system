#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, AveragePooling2D, concatenate, Dense, Activation, Flatten
import copy
import scipy

tf.enable_eager_execution()

# State dataset directory where the tfrecord files are located
dataset_dir = '/Users/hchan/Downloads/CASIA-FaceV5-Crop'

# State where your log file is at. If it doesn't exist, create it.
log_dir = './log'

# State where your checkpoint file is
checkpoint_file = './inception_resnet_v2_2016_08_30.ckpt'

data_root = pathlib.Path(dataset_dir)
piclist = list(data_root.glob('*/*.bmp'))
total_data_size = len(piclist)
batch_size = 32
steps_per_epoch = tf.math.ceil(total_data_size / batch_size)
print("Total image size:", total_data_size)

path_sf1 = copy.deepcopy(piclist)
path_sf2 = copy.deepcopy(piclist)
random.shuffle(path_sf1)
random.shuffle(path_sf2)

path_sf1_str = list(map(str, path_sf1))
path_sf2_str = list(map(str, path_sf2))

# 生成图片label

label_list = [int(pic_path.parent.name) for pic_path in piclist]


def create_dataset():
    for pt1, pt2 in zip(path_sf1_str, path_sf2_str):
        img_raw1 = tf.io.read_file(pt1)
        img_tensor1 = tf.image.decode_bmp(img_raw1, channels=3)
        img_tensor1 = tf.image.resize(img_tensor1, [299, 299])
        img_tensor1 = img_tensor1 / 127.5 - 1
        img_raw2 = tf.io.read_file(pt2)
        img_tensor2 = tf.image.decode_bmp(img_raw2)
        img_tensor2 = tf.image.resize(img_tensor2, [299, 299])
        img_tensor2 = img_tensor2 / 127.5 - 1
        name1 = pt1.split('/')[5]
        name2 = pt2.split('/')[5]
        label = 1 if name1 == name2 else 0
        yield {'input_2': img_tensor1, 'input_3': img_tensor2}, label

def create_dataset_scipy():
    for pt1, pt2 in zip(path_sf1_str, path_sf2_str):
        img_tensor1 = scipy.misc.imread(pt1)
        # scipy.misc.imresize
        img_tensor1 = tf.image.resize(img_tensor1, [299, 299])
        img_tensor1 = img_tensor1 / 127.5 - 1

        img_tensor2 = scipy.misc.imread(pt2)
        img_tensor2 = tf.image.resize(img_tensor2, [299, 299])
        img_tensor2 = img_tensor2 / 127.5 - 1
        name1 = pt1.split('/')[5]
        name2 = pt2.split('/')[5]
        label = 1 if name1 == name2 else 0
        yield {'input_2': img_tensor1, 'input_3': img_tensor2}, label


# 构建Dataset
image_label_ds = tf.data.Dataset.from_generator(create_dataset, ({'input_2': tf.int64, 'input_3': tf.int64},tf.int64))
# image_ds = path_ds.map(create_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)


image_label_ds = image_label_ds.repeat()
image_label_ds = image_label_ds.shuffle(buffer_size=64)
batch_size = 32
steps_per_epoch = tf.math.ceil(total_data_size / batch_size)
image_label_ds = image_label_ds.batch(batch_size)


irv2 = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False)
irv2.trainable = False


# This returns a tensor
input1 = Input(shape=(299, 299, 3))
input2 = Input(shape=(299, 299, 3))

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

model.compile(optimizer='rmsprop', loss='binary_crossentropy')

model.fit_generator(image_label_ds,steps_per_epoch=steps_per_epoch)

# for x, y in image_label_ds:
#     model.fit([x, x], y)
