'''
This file is used for convert the pretrained inception-resnet-v1 model to a SavedModel(Which is used by tensorflow serving)
'''
import os
import hchan.fake_facenet.inception_resnet_v1 as inception_resnet_v1
import tensorflow as tf
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from scipy import misc
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

from hchan.fake_facenet.inception_resnet_v1 import *

EXPORT_PATH = './tf_serving_model/'
MODEL_VERSION = 1

INPUT_SIZE = 160  # the shape of input of inception-resnet-v1 is batch_size,160,160,3
PB_PATH = './pretrained_model/20170512-110547.pb'
GRAPH_META_PATH = './pretrained_model/model-20170512-110547.meta'
CHECKPOINT_FILE = './pretrained_model/model-20170512-110547.ckpt-250000'


def load_graph_from_ckpt():
    saver = tf.train.import_meta_graph(GRAPH_META_PATH)
    # saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), CHECKPOINT_FILE)


def load_graph_from_pb():
    with tf.gfile.GFile(PB_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def)


def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_graph_from_pb()

            #Get placeholders from the graph
            images_placeholder = tf.get_default_graph().get_tensor_by_name("import/input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("import/embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("import/phase_train:0")

            # export model to savedmodel
            export_model_path = os.path.join(EXPORT_PATH,str(MODEL_VERSION))
            print(export_model_path)
            builder = tf.saved_model.builder.SavedModelBuilder(export_model_path)

            tensor_info_x = tf.saved_model.utils.build_tensor_info(images_placeholder)
            tensor_info_train = tf.saved_model.utils.build_tensor_info(phase_train_placeholder)
            tensor_info_y = tf.saved_model.utils.build_tensor_info(embeddings)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': tensor_info_x, 'is_training': tensor_info_train},
                    outputs={'scores': tensor_info_y},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        prediction_signature,
                },
                legacy_init_op=legacy_init_op)

            builder.save()


if __name__ == '__main__':
    main()
