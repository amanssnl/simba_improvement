import tensorflow as tf
import tensorflow_hub as hub
import os
from PIL import Image
from scipy.spatial import distance
import numpy as np
import pandas as pd
from config import metric
from ast import literal_eval
from lion_orientation import lion_orientation_fun
from prepare_face_data import face_extract
from whisker_detection import whisker_roi

# model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
model_url='https://tfhub.dev/google/efficientnet/b7/feature-vector/1'

IMAGE_SHAPE = (224, 224)

layer = hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,))
model = tf.keras.Sequential([layer])

def image_pickup(file):
    file = Image.open(file).convert('L').resize(IMAGE_SHAPE)  # 1
    file = np.stack((file,) * 3, axis=-1)  # 2
    file = np.array(file) / 255.0  # 3

    embedding = model.predict(file[np.newaxis, ])
    # print('embedding_value',embedding)
    embedding_np = np.array(embedding)
    flattended_feature = embedding_np.flatten()
    embedding_list=[]
    for item in flattended_feature:
        embedding_list.append(item)

    return  embedding_list

def distance_match(input_file, file2):
    cosineDistance = distance.cdist([input_file], [file2], metric)[0]
    return cosineDistance[0]


# input_file = 'verify_image/6S2A8718_TDYSAF1.jpg'
# file = Image.open(input_file).convert('L').resize(IMAGE_SHAPE)  #1
# file = np.stack((file,)*3, axis=-1)                       #2
# file = np.array(file)/255.0                               #3
#
# embedding = model.predict(file[np.newaxis, ...])
# embedding_np = np.array(embedding)
# flattended_feature = embedding_np.flatten()
#
#
# input_file1 = 'verify_image/whisker_6S2A8716.jpg'
# file1 = Image.open(input_file1).convert('L').resize(IMAGE_SHAPE)  #1
# file1 = np.stack((file,)*3, axis=-1)                       #2
# file1 = np.array(file)/255.0                               #3
#
# embedding1 = model.predict(file1[np.newaxis, ...])
# embedding_np1 = np.array(embedding1)
# flattended_feature1 = embedding_np1.flatten()
#
# print(distance_match(flattended_feature,flattended_feature1))

    # print(flattended_feature)

# image_pickup('input_dir/whisker_123.jpg')

# #image input for similarity
# input_image='preprocessed_image/test/6S2A1848_JPEG42.jpg'
# face_image_path,tmp_dir=face_extract(input_image)
# test_orientation=lion_orientation_fun(face_image_path)
# whisker_path=whisker_roi(face_image_path,tmp_dir)
#
# csv_filename = 'output/Results_get_data.csv' #embeddind database generated after executing main.py
# df = pd.read_csv(csv_filename)
#
# appended_data=[]
# out_dict={}
# for index, row in df.iterrows():
#     test_image_name=input_image
#     load_embedding_from_df=literal_eval(row['whisker_embedding'])
#     original_image_embedding=np.array(load_embedding_from_df)
#     original_image_orientation=row['face_orientation']
#
#     test_image_embedding=image_pickup(test_image_name)
#     test_image_embedding=np.array(test_image_embedding)
#
#     distance_calculated=distance_match(test_image_embedding,original_image_embedding)
#
#     # print('distance_calculated',distance_calculated)
#     out_dict['existing_lion_name']=row['lion_name']
#     out_dict['existing_image_name']=row['image_name']
#     out_dict['lion_orientation']=row['face_orientation']
#     out_dict['distance_calculated']=distance_calculated
#     out_dict['test_image_name']=test_image_name
#     out_dict['test_image_orientation']=test_orientation
#
#
#     out_df = pd.DataFrame.from_dict([out_dict])
#     appended_data.append(out_df)
#
# appended_data = pd.concat(appended_data, axis = 0)
# appended_data.to_csv('output/final_result.csv',index=False)

# appended_data = []
# for item in input_path:
#     input_file = os.path.join(path_input,item)
#     input_file = Image.open(input_file).convert('L').resize(IMAGE_SHAPE)  #1
#
#     input_file = np.stack((input_file,)*3, axis=-1)                       #2
#     input_file = np.array(input_file)/255.0
#
#     embedding = model.predict(input_file[np.newaxis, ...])
#     embedding_np = np.array(embedding)
#     input_file = embedding_np.flatten()

    # filename = []
    # image_sim = []
    # input_list = []
    # out_dic = {}

    # for file in all_files:
    #     file_name = file
    #     file_path = os.path.join(path,file)
    #     file2 = image_pickup(file_path)
    #     file_name1,cosineDistance = distance_match(input_file, file2, file_name)
    #     filename.append(file_name)
    #     image_sim.append(cosineDistance)
    #     input_list.append(item)
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