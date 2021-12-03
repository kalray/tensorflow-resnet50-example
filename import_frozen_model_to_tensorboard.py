# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Imports a protobuf model as a graph in Tensorboard."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys


from tensorflow.python.framework import ops
from tensorflow.python.framework import importer
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary
from tensorflow.python.client import session
from tensorflow.python.tools import saved_model_utils


def import_pb_to_tensorboard(model_path, log_dir):
    """View an imported protobuf model (`.pb` file) as a graph in Tensorboard.

    Args:
      model_path: The location of the protobuf (`pb`) model to visualize
      log_dir: The location for the Tensorboard log to begin visualization from.

    Usage:
      Call this function with your model location and desired log directory.
      Launch Tensorboard by pointing it to the log directory.
      View your imported `.pb` model as a graph.
    """

    with session.Session(graph=ops.Graph()) as sess:
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = graph_pb2.GraphDef()
            graph_def.ParseFromString(f.read())
            importer.import_graph_def(graph_def)
        pb_visual_writer = summary.FileWriter(log_dir)
        pb_visual_writer.add_graph(graph_def)

    print("-----------------------------------------")
    print("Model {} Imported. Visualize by running: "
              "tensorboard --logdir={}".format(model_path, log_dir))
    print("-----------------------------------------")
    return graph_def


def main(unused_args):
    import_pb_to_tensorboard(FLAGS.model_path, FLAGS.log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The location of the protobuf (\'pb\') model to visualize.")
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="The location for the Tensorboard log to begin visualization from.")

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
