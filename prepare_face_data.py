import os
import cv2
import shutil
import tempfile
import pandas as pd
from PIL import Image
from utils import lion_model, extract_lion_data

def face_extract(lion_image,filename):
    tmp_dir = None
    face_path =''
    theta=0
    # filename=filename.split('.')[0]
    try:
        # tmp_dir = tempfile.mkdtemp()
        prefix=r'C:\Users\amans\PycharmProjects\testing_pipeline\processed'
        path=os.path.join(prefix,filename)
        os.makedirs(path)
        tmp_dir=path
        # print("tmp_dir:",tmp_dir)

        pil_img = Image.open(lion_image)
        src = cv2.imread(lion_image)
        temp_image = src.copy()
        coordinates, whisker_cords, face_cords, status = lion_model.get_coordinates(lion_image,
                                                                                    'temp_lion')
        print('face_cords prepare_face_data: ',face_cords)
        if status != "Success":
            pass
        lion_path, face_path,theta= extract_lion_data(face_cords, 'temp_lion', pil_img, coordinates, tmp_dir, temp_image)

    except Exception as e:
        import traceback
        traceback.print_exc()
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    return face_path,tmp_dir,theta

# face_extract('input_dir/6S2A3400_JJPEG4.jpg','6S2A3400_JJPEG4')