import os
from tensorflow import keras
import numpy as np
import pandas as pd
from lion_orientation import lion_orientation_fun
# from image_matching_effi import image_pickup
from prepare_face_data import face_extract
from whisker_detection import whisker_roi
from yolo_whisker_detection import predict_whisker_yolo
from similarity_efffi import image_pickup

try:
    #creating empty list to save orientation and image name
    appended_data=[]
    out_dict = {}

    source_dir = 'preprocessed_image/onboard'

    lion_name_dirs = os.listdir(source_dir)
    for lion_name_dir in lion_name_dirs:
        root_dir = os.path.join(source_dir, lion_name_dir)
        lion_images = os.listdir(root_dir)
        lion_names = []
        image_name = []
        orientation = []
        whisker_embedding = []

        for lion_image in lion_images:

            filename= lion_image
            folder_name=lion_name_dir
            image_path=os.path.join(root_dir, lion_image)

            face_path,tmp_dir,theta = face_extract(image_path,filename)
            print("face_path_retutn main line 35:",face_path)
            if (face_path =='') or (face_path is None) :
                pass
            else:
                final_orientation = lion_orientation_fun(face_path)

                print("orienataiton is :::::;",final_orientation)
                #whisker extract and save roi to tmp_dir
                # whisker_path=whisker_roi(image_path,tmp_dir,theta)
                whisker_path = predict_whisker_yolo(face_path,tmp_dir,theta)
                if whisker_path == '':
                    pass
                else:
                    print("whisker path is :---", whisker_path)

                    individual_embedding = image_pickup(whisker_path)

                    # print("embedding:",individual_embedding)

                    lion_names.append(folder_name)
                    image_name.append(filename)
                    orientation.append(final_orientation)
                    whisker_embedding.append(individual_embedding)

            # storing list data to dict for generating csv files
        out_dict['lion_name'] = lion_names
        out_dict['image_name'] = image_name
        out_dict['face_orientation'] = orientation
        out_dict['whisker_embedding'] = whisker_embedding


        out_df = pd.DataFrame.from_dict(out_dict)
        appended_data.append(out_df)


    appended_data = pd.concat(appended_data, axis = 0)
    appended_data.to_csv('output/onboard_db_yolo.csv',index=False)

except Exception as e:
    import traceback
    traceback.print_exc()




