from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import resnet50
from keras.applications.imagenet_utils import decode_predictions
import keras as keras
import numpy as np

model = keras.applications.resnet50.ResNet50(weights="imagenet")
path = "../input/starfish/asd.jpg"
# load an image in PIL format
original_image = load_img(path, target_size=(224, 224))
numpy_image = img_to_array(original_image)

# Convert the image into 4D Tensor (samples, height, width, channels) by adding an extra dimension to the axis 0.
input_image = np.expand_dims(numpy_image, axis=0)

# preprocess for resnet50
processed_image_resnet50 = resnet50.preprocess_input(input_image.copy())

# resnet50
predictions_resnet50 = model.predict(processed_image_resnet50)
label_resnet50 = decode_predictions(predictions_resnet50)
print (label_resnet50)