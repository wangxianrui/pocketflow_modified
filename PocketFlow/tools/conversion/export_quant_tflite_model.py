# Tencent is pleased to support the open source community by making PocketFlow available.
#
# Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Export a quantized *.tflite model from checkpoint files."""

import os
import re
import traceback
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow.contrib import graph_editor
from tensorflow.contrib.lite.python import lite_constants as constants

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('log_dir', './logs', 'logging directory')
tf.app.flags.DEFINE_string('model_dir', './models_uqtf_eval', 'model directory')
tf.app.flags.DEFINE_string('input_coll', 'images_final', 'input tensor\'s collection')
tf.app.flags.DEFINE_string('output_coll', 'logits_final', 'output tensor\'s collection')
tf.app.flags.DEFINE_boolean('enbl_post_quant', False, 'enable post-training quantization')


def get_file_path_meta():
    """Get the file path to the *.meta data.

    Returns:
    * file_path: file path to the *.meta data
    """

    pattern = re.compile(r'model\.ckpt\.meta$')
    for file_name in os.listdir(FLAGS.model_dir):
        if re.search(pattern, file_name) is not None:
            file_path = os.path.join(FLAGS.model_dir, file_name)
            break

    return file_path


def get_input_name_n_shape(file_path):
    """Get the input tensor's name & shape from *.meta file.

    Args:
    * file_path: file path to the *.meta data

    Returns:
    * input_name: input tensor's name
    * input_shape: input tensor's shape
    """

    with tf.Graph().as_default():
        tf.train.import_meta_graph(file_path)
        net_input = tf.get_collection(FLAGS.input_coll)[0]
        input_name = net_input.name
        input_shape = net_input.shape

    return input_name, input_shape


def convert_pb_model_to_tflite(net, file_path_pb, file_path_tflite):
    """Convert *.pb model to a *.tflite model.

    Args:
    * net: network configurations
    * file_path_pb: file path to the *.pb model
    * file_path_tflite: file path to the *.tflite model
    * enbl_quant: whether to enable quantization
    """
    #
    # tf.logging.info(file_path_pb + ' -> ' + file_path_tflite)
    # arg_list = [
    #     '--graph_def_file ' + file_path_pb,
    #     '--output_file ' + file_path_tflite,
    #     '--input_arrays ' + net['input_name'],
    #     '--output_arrays ' + net['output_name']]
    # if not enbl_quant:
    #     arg_list += ['--inference_type FLOAT']
    # else:
    #     arg_list += [
    #         '--inference_type QUANTIZED_UINT8',
    #         '--mean_values 128',
    #         '--std_dev_values 127']
    #     if FLAGS.enbl_post_quant:
    #         arg_list += [
    #             '--default_ranges_min 0',
    #             '--default_ranges_max 6']
    # cmd_str = ' '.join(['tflite_convert'] + arg_list)
    # subprocess.call(cmd_str, shell=True)
    # tf.logging.info(file_path_tflite + ' generated')

    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(file_path_pb, [net['input_name']],
                                                                  [net['output_name']])

    tf.logging.info(file_path_pb + ' -> ' + file_path_tflite['original'])
    original_tflite = converter.convert()
    with tf.gfile.GFile(file_path_tflite['original'], 'wb') as file:
        file.write(original_tflite)
    tf.logging.info(file_path_tflite['original'] + ' generated')

    tf.logging.info(file_path_pb + ' -> ' + file_path_tflite['post_quant'])
    converter.post_training_quantize = True
    post_quant_tflite = converter.convert()
    with tf.gfile.GFile(file_path_tflite['post_quant'], 'wb') as file:
        file.write(post_quant_tflite)
    tf.logging.info(file_path_tflite['post_quant'] + ' generated')

    tf.logging.info(file_path_pb + ' -> ' + file_path_tflite['infer_quant'])
    converter.inference_type = constants.QUANTIZED_UINT8
    converter.quantized_input_stats = {net['input_name']: {128., 127.}}
    if FLAGS.enbl_post_quant:
        converter.default_ranges_stats = (0, 6)
    infer_quant_tflite = converter.convert()
    with tf.gfile.GFile(file_path_tflite['infer_quant'], 'wb') as file:
        file.write(infer_quant_tflite)
    tf.logging.info(file_path_tflite['infer_quant'] + ' generated')


def test_pb_model(file_path, net_input_name, net_output_name, net_input_data):
    """Test the *.pb model.

    Args:
    * file_path: file path to the *.pb model
    * net_input_name: network's input node's name
    * net_output_name: network's output node's name
    * net_input_data: network's input node's data
    """

    with tf.Graph().as_default() as graph:
        sess = tf.Session()

        # restore the model
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(file_path, 'rb') as i_file:
            graph_def.ParseFromString(i_file.read())
        tf.import_graph_def(graph_def)

        # obtain input & output nodes and then test the model
        net_input = graph.get_tensor_by_name('import/' + net_input_name + ':0')
        net_output = graph.get_tensor_by_name('import/' + net_output_name + ':0')
        tf.logging.info('input: {} / output: {}'.format(net_input.name, net_output.name))
        net_output_data = sess.run(net_output, feed_dict={net_input: net_input_data})
        tf.logging.info('outputs from the *.pb model: {}'.format(net_output_data))


def test_tflite_model(file_path, net_input_data):
    """Test the *.tflite model.

    Args:
    * file_path: file path to the *.tflite model
    * net_input_data: network's input node's data
    """

    # restore the model and allocate tensors
    interpreter = tf.contrib.lite.Interpreter(model_path=file_path)
    interpreter.allocate_tensors()

    # get input & output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tf.logging.info('input details: {}'.format(input_details))
    tf.logging.info('output details: {}'.format(output_details))

    # test the model with given inputs
    interpreter.set_tensor(input_details[0]['index'], net_input_data)
    interpreter.invoke()
    net_output_data = interpreter.get_tensor(output_details[0]['index'])
    tf.logging.info('outputs from the *.tflite model: {}'.format(net_output_data))


def is_initialized(sess, var):
    """Check whether a variable is initialized.

    Args:
    * sess: TensorFlow session
    * var: variabile to be checked
    """

    try:
        sess.run(var)
        return True
    except tf.errors.FailedPreconditionError:
        return False


def replace_dropout_layers():
    """Replace dropout layers with identity mappings.

    Returns:
    * op_outputs_old: output nodes to be swapped in the old graph
    * op_outputs_new: output nodes to be swapped in the new graph
    """

    pattern_div = re.compile('/dropout/div')
    pattern_mul = re.compile('/dropout/mul')
    op_outputs_old, op_outputs_new = [], []
    for op in tf.get_default_graph().get_operations():
        if re.search(pattern_div, op.name) is not None:
            x = tf.identity(op.inputs[0])
            op_outputs_new += [x]
        if re.search(pattern_mul, op.name) is not None:
            op_outputs_old += [op.outputs[0]]

    return op_outputs_old, op_outputs_new


def export_pb_tflite_model(net, file_path_meta, file_path_pb, file_paths_tflite):
    """Export *.pb & *.tflite models from checkpoint files.

    Args:
    * net: network configurations
    * file_path_meta: file path to the *.meta data
    * file_path_pb: file path to the *.pb model
    * file_paths_tflite: dictionary of file paths to *.tflite models
    """

    # convert checkpoint files to a *.pb model
    with tf.Graph().as_default() as graph:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # pylint: disable=no-member
        sess = tf.Session(config=config)

        # restore the graph with inputs replaced
        net_input = tf.placeholder(tf.float32, shape=net['input_shape'], name=net['input_name'])
        saver = tf.train.import_meta_graph(
            file_path_meta, input_map={net['input_name_ckpt']: net_input})
        saver.restore(sess, file_path_meta.replace('.meta', ''))

        # obtain the output node
        net_logits = tf.get_collection(FLAGS.output_coll)[0]
        net_output = tf.nn.softmax(net_logits, name=net['output_name'])
        tf.logging.info('input: {} / output: {}'.format(net_input.name, net_output.name))
        tf.logging.info('input\'s shape: {}'.format(net_input.shape))
        tf.logging.info('output\'s shape: {}'.format(net_output.shape))

        # replace dropout layers with identity mappings (TF-Lite does not support dropout layers)
        op_outputs_old, op_outputs_new = replace_dropout_layers()
        sess.close()
        graph_editor.swap_outputs(op_outputs_old, op_outputs_new)
        sess = tf.Session(config=config)  # open a new session
        saver.restore(sess, file_path_meta.replace('.meta', ''))

        # write the original grpah to *.pb file
        graph_def = graph.as_graph_def()
        graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [net['output_name']])
        file_name_pb = os.path.basename(file_path_pb)
        tf.train.write_graph(graph_def, FLAGS.model_dir, file_name_pb, as_text=False)
        tf.logging.info(file_path_pb + ' generated')

    # convert the *.pb model to a *.tflite model
    # convert_pb_model_to_tflite(net, file_path_pb, file_paths_tflite['float'], enbl_quant=False)
    # convert_pb_model_to_tflite(net, file_path_pb, file_paths_tflite['quant'], enbl_quant=True)
    convert_pb_model_to_tflite(net, file_path_pb, file_paths_tflite)

    # test *.pb & *.tflite models
    tf.logging.info('test pb file')
    test_pb_model(file_path_pb, net['input_name'], net['output_name'], net['input_data'])
    tf.logging.info('test original tflite file')
    test_tflite_model(file_paths_tflite['original'], net['input_data'])
    tf.logging.info('test post_quant tflite file')
    test_tflite_model(file_paths_tflite['post_quant'], net['input_data'])
    tf.logging.info('test infer_quant tflite file')
    test_tflite_model(file_paths_tflite['infer_quant'], net['input_data'].astype(np.uint8))


def main(unused_argv):
    """Main entry.

    Args:
    * unused_argv: unused arguments (after FLAGS is parsed)
    """

    try:
        # setup the TF logging routine
        tf.logging.set_verbosity(tf.logging.INFO)

        # network configurations
        file_path_meta = get_file_path_meta()
        input_name, input_shape = get_input_name_n_shape(file_path_meta)
        net = {
            'input_name_ckpt': input_name,  # used to import the model from checkpoint files
            'input_name': 'net_input',  # used to export the model to *.pb & *.tflite files
            'input_shape': input_shape,
            'output_name': 'net_output'
        }
        net['input_data'] = np.zeros(tuple([1] + list(net['input_shape'])[1:]), dtype=np.float32)

        # generate *.pb & *.tflite files
        file_path_pb = os.path.join(FLAGS.model_dir, 'model_original.pb')
        file_paths_tflite = {'original': os.path.join(FLAGS.model_dir, 'model_original.tflite'),
                             'post_quant': os.path.join(FLAGS.model_dir, 'model_post_quantized.tflite'),
                             'infer_quant': os.path.join(FLAGS.model_dir, 'model_infer_quantized.tflite')
                             }
        export_pb_tflite_model(net, file_path_meta, file_path_pb, file_paths_tflite)

        # exit normally
        return 0
    except ValueError:
        traceback.print_exc()
        return 1  # exit with errors


if __name__ == '__main__':
    tf.app.run()
