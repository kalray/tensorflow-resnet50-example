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
import sys
import time
import yaml
import shutil
import subprocess
import collections
import numpy as np
from functools import reduce

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


def array_from_fifo(fd, dtype, count):
    arr = np.empty(count, dtype=dtype)
    nb_read = fd.readinto(memoryview(arr))
    if nb_read < arr.nbytes:
        raise Exception("Read failed, EOF or pipe closed")
    return arr


def read_kann_output(kann_out):
    # Ordered to keep the alphabetical order
    data = collections.OrderedDict()
    for name, output in kann_out.items():
        file = output['fifo']
        size = output['size']
        try:
            data[name] = array_from_fifo(file, dtype=np.float32, count=size)
        except:
            raise Exception("Reading of {} values from {} failed"
                .format(size, name))
    return data


def run_kann_inference(image_dir, fifos_in, fifos_out, input_name, output_name,
                       input_shape, output_shape, batch_size=1):

    if batch_size is None:
        batch_size = input_shape[0][1]
    kann_in = collections.OrderedDict()
    kann_out = collections.OrderedDict()
    buffers = sorted(input_name + output_name)
    for b in buffers:
        if b in input_name:
            print("Opening input fifo for CNN's input : '{}'".format(b))
            kann_in[b] = {'fifo': os.fdopen(os.open(fifos_in[b], os.O_WRONLY), 'wb', 0)}
    for b in buffers:
        if b in output_name:
            print("Opening output fifo for CNN's output : '{}'".format(b))
            kann_out[b] = {'fifo': os.fdopen(os.open(fifos_out[b], os.O_RDONLY), 'rb', 0)}
    for b, shape in zip(output_name, output_shape):
        kann_out[b]['size'] = reduce(lambda x, y: x * y, shape)

    for image_file in os.listdir(image_dir):
        print(f'prediction of {image_file}')
        print('---------------------------')

        img_path = os.path.join(image_dir, image_file)
        img = image.load_img(img_path, target_size=(input_shape[0][0], input_shape[0][2]))
        img_prepared = image.img_to_array(img)
        img_prepared = np.expand_dims(img_prepared, axis=0)  # batch
        # img_prepared = np.array([img_prepared] * batch_size, dtype=img_prepared.dtype)
        img_prepared = preprocess_input(img_prepared)
        img_prepared = [img_prepared.transpose((1, 0, 2, 3))]
        for p, i in zip(img_prepared, kann_in.values()):
            p.tofile(i['fifo'], '')

        s = time.time()
        out = read_kann_output(kann_out)
        t = time.time() - s
        fps = 1 / t
        for preds, values in out.items():
            for (_, class_predicted, prob) in decode_predictions(np.expand_dims(values, axis=0), top=3)[0]:
                print('- class: {}({:.2f}%)'.format(class_predicted, 100 * prob))
        print(f'time prediction {1e3 * t:.3f} ms - {fps:.2f} FPS')
        print('---------------------------')


def main(data_dir_path, model_dir):

    kann_bin_path = [d for d in os.listdir(model_dir) if d.split('.')[-1] == 'bin'][0]
    kann_bin_path = os.path.join(model_dir, kann_bin_path)
    kann_cfg_file_path = os.path.join(model_dir, 'network.dump.yaml')
    io_model_path = 'io'

    cfg_path = os.path.join(os.getcwd(), kann_cfg_file_path)
    with open(cfg_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    print(f"input: {config_dict['input_nodes_name'][0]}\t{config_dict['input_nodes_shape'][0]}")
    print(f"output: {config_dict['output_nodes_name'][0]}\t{config_dict['output_nodes_shape'][0]}")

    fifos_dir = io_model_path
    kann_proc = None

    print("Directory for the fifos is {}".format(fifos_dir))
    if os.path.exists(fifos_dir):
        shutil.rmtree(fifos_dir)
    fifos_in = {}
    for input_ in config_dict['input_nodes_name']:
        input_path = fifos_dir + "/{}".format(input_)
        dir = os.path.dirname(input_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        os.mkfifo(input_path)
        fifos_in[input_] = input_path
    fifos_out = {}
    for output in config_dict['output_nodes_name']:
        output_path = fifos_dir + "/{}".format(output)
        dir = os.path.dirname(output_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        os.mkfifo(output_path)
        fifos_out[output] = output_path

    logfile_path = os.path.join(os.getcwd(), 'inference.log')
    flog = open(logfile_path, 'w+')
    kann_proc = subprocess.Popen(['kann_opencl_cnn', '.',
                                 kann_bin_path, fifos_dir], bufsize=-1,
                                 stdout=flog,
                                )

    run_kann_inference(data_dir_path, fifos_in, fifos_out,
                       config_dict['input_nodes_name'], config_dict['output_nodes_name'],
                       config_dict['input_nodes_shape'], config_dict['output_nodes_shape'],
                       config_dict['batch_size'],
                       )

    kann_proc.wait(timeout=5)
    print("Killing KaNN(TM) process")
    kann_proc.terminate()
    flog.close()


if __name__ == '__main__':

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        print("Please provide a generated model with KaNN(tm)")
        sys.exit(1)
    data_path = sys.argv[2] if len(sys.argv) == 3 else r'images'
    main(data_path, model_path)
