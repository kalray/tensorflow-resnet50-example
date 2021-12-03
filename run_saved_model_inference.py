import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


def run_inference(model, image_dir):
    for image_file in os.listdir(image_dir):

        print(f'prediction of {image_file}')
        print('---------------------------')
        img_path = os.path.join(image_dir, image_file)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) # batch
        x = preprocess_input(x)
        s = time.time()
        preds = model.predict(x)
        t = time.time() - s
        fps = 1 / t
        # decode the results into a list of tuples (class, description, probability)
        # (one such list for each sample in the batch)
        for (_, class_predicted, prob) in decode_predictions(preds, top=3)[0]:
            print('- class: {}({:.2f}%)'.format(class_predicted, 100 * prob))
        print(f'time prediction {1e3 * t:.3f} ms - {fps:.2f} FPS')
        print('---------------------------')


model_path = r'saved_model'
model = tf.keras.models.load_model(model_path)
run_inference(model, r'images')
