

from keras.preprocessing import image
from tensorflow import keras
import numpy as np
import os
orientation_model= os.path.join('models','lion_orientation.h5')
model = keras.models.load_model(orientation_model)

def lion_orientation_fun(lion_path): # thsi function return only one argument that is "lion orientation'

    test_image = image.load_img(lion_path, color_mode ='rgb',
                                target_size = (224, 224))


    test_image = image.img_to_array(test_image,dtype = "uint8")
    test_image = np.expand_dims(test_image, axis= 0)
    result = model.predict(test_image)
    # print(result)
    # print(type(result))
    res = np.argmax(result)

    dict1 = {0 : 'front', 1: 'left', 2: 'right'}
    return dict1[res]
    # print("The predicted output is :",dict1[res])

# test_image='processed/6S2A8819/rotated.jpg'
# orientation_result=lion_orientation_fun(test_image)
# print(orientation_result)