from tensorflow.keras.applications.resnet50 import ResNet50


model_path = r'saved_model'
model = ResNet50(weights='imagenet')
model.save(model_path)
