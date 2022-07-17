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
from similarity_efffi import image_pickup,distance_match
from whisker_detection import whisker_roi


# # model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
# model_url='https://tfhub.dev/google/efficientnet/b7/feature-vector/1'
# #
# IMAGE_SHAPE = (224, 224)
# #
# layer = hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,))
# model = tf.keras.Sequential([layer])


#image input for similarity
input_image='preprocessed_image/test/6S2A2031_JPEG52_1.jpg'
image_name=input_image.split('.')[0].split('/')[-1]
face_image_path,tmp_dir,theta=face_extract(input_image,image_name)
test_orientation=lion_orientation_fun(face_image_path)
whisker_path=whisker_roi(face_image_path,tmp_dir,theta)

csv_filename = 'output/onboard_db.csv' #embeddind database generated after executing main.py
df = pd.read_csv(csv_filename)

appended_data=[]
out_dict={}
for index, row in df.iterrows():
    test_image_name=input_image
    load_embedding_from_df=literal_eval(row['whisker_embedding'])
    original_image_embedding=np.array(load_embedding_from_df)
    original_image_orientation=row['face_orientation']

    print("image  is getting processed:", row['image_name'])
    if (test_orientation=='front'):
        distance_calculated=-9999
    else:
        if (test_orientation==original_image_orientation):
            test_image_embedding=image_pickup(test_image_name)
            test_image_embedding=np.array(test_image_embedding)

            distance_calculated=distance_match(test_image_embedding,original_image_embedding)
        else:
            distance_calculated=-9999
    # print('distance_calculated',distance_calculated)
    out_dict['existing_lion_name']=row['lion_name']
    out_dict['existing_image_name']=row['image_name']
    out_dict['lion_orientation']=row['face_orientation']
    out_dict['distance_calculated']=distance_calculated
    out_dict['test_image_name']=image_name
    out_dict['test_image_orientation']=test_orientation


    out_df = pd.DataFrame.from_dict([out_dict])
    appended_data.append(out_df)


appended_data = pd.concat(appended_data, axis = 0)
appended_data.to_csv('output/final_result_6S2A2031_JPEG52_1.csv',index=False)

