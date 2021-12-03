import os
import cv2
import glob
import numpy
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


IMAGE_WIDTH, IMAGE_HEIGHT = (224, 224)


# We need to provide a generator that give a representative dataset of inputs
def representative_dataset_gen():
    dataset_name = 'ILSVRC2012_VAL'
    dataset_path = r"../dataset/{}/images/".format(dataset_name)
    images = glob.glob(os.path.join(dataset_path, '*.JPEG'))
    for i in range(len(os.listdir(dataset_path))):
        if i % (len(os.listdir(dataset_path)) / 10) == 0:
            print('Building representative dataset on \t {} / {} images ({:.1f} %)'.format(
                i, len(os.listdir(dataset_path)), 100 * i / len(os.listdir(dataset_path))))
        img = image.load_img(images[i], target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
        img = image.img_to_array(img)
        img = numpy.expand_dims(img, axis=0)  # batch
        img = preprocess_input(img)
        yield [img]
    print('Building representative dataset on \t {} / {} images ({:.1f} %)'.format(
        i, len(os.listdir(dataset_path)), 100 * i / len(os.listdir(dataset_path))))
    print('done')


# Define variables input / output
saved_model_dir = r'saved_model'
dest_dir_path = r'tflite_model'
dest_model_path = os.path.join(dest_dir_path, 'frozen.model.tflite')

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model_quant = converter.convert()

# Save the model
if not os.path.exists(dest_dir_path):
    os.makedirs(dest_dir_path)
with open(dest_model_path, 'wb') as f:
    f.write(tflite_model_quant)
print('TFlite model written at {}'.format(dest_model_path))

