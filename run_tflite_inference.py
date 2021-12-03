# Copyright 2021 The Kalray Authors. All Rights Reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


def get_io_name(model_path, node_in_op='Placeholder', node_out_op='Softmax'):
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = graph_pb2.GraphDef()
        graph_def.ParseFromString(f.read())

        inputs_name = [node.name for node in graph_def.node if node.op == node_in_op]
        outputs_name = [node.name for node in graph_def.node if node.op == node_out_op]

        if len(inputs_name) > 0 and len(outputs_name) > 0:
            print(f'Found inputs ({node_in_op}): {inputs_name}')
            print(f'Found outputs ({node_out_op}): {outputs_name}')
            return inputs_name, outputs_name
        else:
            raise IOError(f'Inputs or Outputs not found - in: {inputs_name} | out {outputs_name}')


def run_inference(model_path, image_dir):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input = interpreter.get_input_details()
    output = interpreter.get_output_details()
    input_shape = input[0]['shape']

    for image_file in os.listdir(image_dir):
        print(f'prediction of {image_file}')
        print('---------------------------')
        img_path = os.path.join(image_dir, image_file)
        img = image.load_img(img_path, target_size=(input_shape[1], input_shape[2]))
        input_data = image.img_to_array(img)
        input_data = np.expand_dims(input_data, axis=0)  # batch
        input_data = preprocess_input(input_data)
        interpreter.set_tensor(input[0]['index'], input_data)
        # Get predictions (or output:))
        s = time.time()
        interpreter.invoke()
        t = time.time() - s
        fps = 1 / t
        output_data = interpreter.get_tensor(output[0]['index'])
        for (_, class_predicted, prob) in decode_predictions(output_data, top=3)[0]:
            print('- class: {}({:.2f}%)'.format(class_predicted, 100 * prob))
        print(f'time prediction {1e3 * t:.3f} ms - {fps:.2f} FPS')
        print('---------------------------')


tflite_model_path = 'tflite_model/frozen.model.tflite'
data_dir_path = r'images'
saved_model_path = 'saved_model'

run_inference(tflite_model_path, data_dir_path)
