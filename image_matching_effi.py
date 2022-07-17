import tensorflow as tf
import tensorflow_hub as hub
import os
from PIL import Image
from scipy.spatial import distance
import numpy as np
import pandas as pd
from config import path, path_input, metric, model_url

#
# all_files = os.listdir(path)   # extract this data from csv file
# input_path = os.listdir(path_input)

IMAGE_SHAPE = (224, 224)

layer = hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,))
model = tf.keras.Sequential([layer])

def image_pickup(file2):

    file2 = Image.open(file2).convert('L').resize(IMAGE_SHAPE)  #1
    # file2 = ImageOps.grayscale(file2)
    # file2.show()
    file2 = np.stack((file2,)*3, axis=-1)                       #2
    file2 = np.array(file2)/255.0                               #3
    embedding = model.predict(file2[np.newaxis, ...])
    embedding_np = np.array(embedding)
    file2 = embedding_np.flatten()
    return file2

def distance_match(input_file, file2, file_name1):
    cosineDistance = distance.cdist([input_file], [file2], metric)[0]
    return file_name1, cosineDistance[0]

#
# appended_data = []
# for item in input_path:
#     input_file = os.path.join(path_input,item)
#     # input_file = ImageOps.grayscale(input_file)
#     print(item)
#
#     input_file = Image.open(input_file).convert('L').resize(IMAGE_SHAPE)  #1
#     # input_file.show()
#     input_file = np.stack((input_file,)*3, axis=-1)                       #2
#     input_file = np.array(input_file)/255.0
#
#     embedding = model.predict(input_file[np.newaxis, ...])
#     embedding_np = np.array(embedding)
#     input_file = embedding_np.flatten()
#
#     filename = []
#     image_sim = []
#     input_list = []
#     out_dic = {}
#
#     for file in all_files:
#         file_name = file
#         file_path = os.path.join(path,file)
#         file2 = image_pickup(file_path)
#         file_name1,cosineDistance = distance_match(input_file, file2, file_name)
#         filename.append(file_name)
#         image_sim.append(cosineDistance)
#         input_list.append(item)
#
#     out_dic['filename'] = filename
#     out_dic['distance'] = image_sim
#     out_dic['input_file'] = input_list
#
#     out_df = pd.DataFrame.from_dict(out_dic)
#     appended_data.append(out_df)
#
# appended_data = pd.concat(appended_data, axis = 1)
# appended_data.to_csv(r'Results\Results_sc_left.csv')