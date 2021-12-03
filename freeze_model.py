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
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


class ModelInput(object):
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype


model_path = r'saved_model/'
freeze_path = r'./frozen_model/frozen.model.pb'

input = ModelInput(name='image', shape=(1, 224, 224, 3), dtype=tf.float32)

model = tf.keras.models.load_model(model_path)
model.trainable = False
model.summary()

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x, training=False))
full_model = full_model.get_concrete_function({
    'input_1': tf.TensorSpec(input.shape, input.dtype, name=input.name)
})

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)

print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=os.path.dirname(freeze_path),
                  name=os.path.basename(freeze_path),
                  as_text=False)
print(f'Model has been frozen to {freeze_path}')
