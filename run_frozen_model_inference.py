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
from tensorflow.python.client import session
from tensorflow.python.platform import gfile
from tensorflow.python.framework import ops
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


def run_inference(model_path, image_dir, input_name, output_name):
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    with session.Session(graph=ops.Graph(), config=session_conf) as sess:
        # Import freeze model
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = graph_pb2.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        graph = tf.get_default_graph()
        # Get output tensor
        detections = graph.get_tensor_by_name(f'{output_name}:0')
        outputs = {f'{output_name}': detections}

        for image_file in os.listdir(image_dir):
            print(f'prediction of {image_file}')
            print('---------------------------')
            img_path = os.path.join(image_dir, image_file)
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)  # batch
            x = preprocess_input(x)
            # Associate to each placeholder the value(numpy array) to feed into the graph
            feed = {f'{input_name}:0': x}
            # Get predictions (or output:))
            s = time.time()
            preds = sess.run(outputs, feed_dict=feed)
            t = time.time() - s
            fps = 1 / t
            for (_, class_predicted, prob) in decode_predictions(preds[output_name], top=3)[0]:
                print('- class: {}({:.2f}%)'.format(class_predicted, 100 * prob))
            print(f'time prediction {1e3 * t:.3f} ms - {fps:.2f} FPS')
            print('---------------------------')


frozen_model_path = 'frozen_model/frozen.model.pb'
data_dir_path = r'images'

input_name, output_name = get_io_name(frozen_model_path)
run_inference(frozen_model_path, data_dir_path, input_name[0], output_name[0])
