import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
import math

import cv2
import numpy as np
from PIL import Image
#import image
import logging
import tensorflow as tf
from skimage.transform import resize
from keras.models import load_model

from lion_model import LionDetection, classes
from train_utils import read_and_resize, augment
from keras.applications.resnet50 import preprocess_input
from lion_orientation import lion_orientation_fun
# from compressed_Table import insert_compressed_data, img_hash_value, duplicate_img_detected
# from whisker_spot import predict_whisker, whisker_spot_calculation

cwd = os.getcwd()


import imagehash

lion_model = LionDetection()


print("Model Init Done!")


def current_milli_time():
    return round(time.time() * 1000)


def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def ratio_calculate(left_area, right_area):
    face_orientation='U'
    ratio=0
    print("left_area:",left_area)
    print("right_araea",right_area)
    if (left_area == 0) and (right_area == 0):
        face_orientation='U'
        ratio=-9999
    else:
        if (left_area == 0) or (right_area == 0):
            ratio=9999
            if left_area==0:
                face_orientation='R'
            else:
                face_orientation='L'
        else:
            ratio=left_area/right_area
            if (0.8 <= ratio <= 1.2):
                face_orientation='F'
            else:
                if ratio>1:
                    face_orientation='L'
                else:
                    face_orientation='R'

    print("ratio:{} and face_orientation:{}".format(ratio, face_orientation))
    return ratio,face_orientation




def load_and_align_images(images):
    aligned_images = []
    for image in images:
        cropped = image
        image_size = 160

        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)

    return np.array(aligned_images)


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def calculate_face_embeddings(image):
    image = read_and_resize(image)
    image = augment(image)
    image = preprocess_input(np.array(image))
    image = np.array([image])
    embeddings = keras_face_model.predict(image)
    return embeddings


def calculate_whisker_embeddings(image):
    image = read_and_resize(image)
    image = augment(image)
    image = preprocess_input(np.array(image))
    image = np.array([image])
    embeddings = keras_whisker_model.predict(image)
    return embeddings


def extract_lion_data(face_cords, lion, pil_img, coordinates, tmp_dir, temp_image):
    lion_path = ''
    face_path = ''
    nose_path = ''
    leye_path = ''
    reye_path = ''
    theta=0

    for face_coord in face_cords[lion]['boxes']:
        if face_coord["conf"] > 0.7:
            _coordinates = []
            face = pil_img.copy()
            nose = pil_img.copy()
            leye = pil_img.copy()
            reye = pil_img.copy()
            print("line 137 utils")


            for coord in coordinates['boxes']:
                if lion_model.insideface(face_coord, coord):
                    _coordinates.append(coord)
                    roi_box = coord['ROI']
                    xmin = int(roi_box[0])
                    ymin = int(roi_box[1])
                    xmax = int(roi_box[2])
                    ymax = int(roi_box[3])
                    temp_image = cv2.rectangle(temp_image,
                                               (xmin, ymin),
                                               (xmax, ymax),
                                               (36, 255, 12),
                                               4)
                    cv2.putText(temp_image,
                                classes[str(coord['class'])],
                                (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_PLAIN,
                                4,
                                (36, 255, 12),
                                2)
                    if coord['class'] in [1, 2, 3, 4, 5]:
                        face = face.crop((xmin, ymin, xmax, ymax,))
                        face_path = os.path.join(tmp_dir, "face.jpg")
                        print("face path utils line 162:",face_path)
                        face.save(face_path)
                        # face_arr = cv2.imread(face_path)
                        # face_emb = calculate_face_embeddings(face_path)
                        # face_str_embedding = [str(a) for a in list(face_emb[0])]
                        # face_embedding = ','.join(face_str_embedding)
                    # elif coord['class'] in [27, 28, 29, 30, 31]:
                    #     whisker = whisker.crop((xmin, ymin, xmax, ymax,))
                    #     whisker_path = os.path.join(tmp_dir, "whisker.jpg")
                    #     whisker.save(whisker_path)
                    #     # whisker_arr = cv2.imread(whisker_path)
                    #     # whisker_emb = calculate_whisker_embeddings(whisker_path)
                    #     # whisker_str_embedding = [str(a) for a in list(whisker_emb[0])]
                    #     # whisker_embedding = ','.join(whisker_str_embedding)
                    # elif coord['class'] in [6, 8, 10, 12]:
                    #     lear = lear.crop((xmin, ymin, xmax, ymax,))
                    #     lear_path = os.path.join(tmp_dir, "lear.jpg")
                    #     lear.save(lear_path)
                    # elif coord['class'] in [7, 9, 11, 13]:
                    #     rear = rear.crop((xmin, ymin, xmax, ymax,))
                    #     rear_path = os.path.join(tmp_dir, "rear.jpg")
                    #     rear.save(rear_path)
                    elif coord['class'] in [14, 16, 18, 20]:
                        leye = leye.crop((xmin, ymin, xmax, ymax,))
                        center_leye_roi_x, center_leye_roi_y = ((xmin + xmax) / 2), ((ymin + ymax) / 2)
                        leye_path = os.path.join(tmp_dir, "leye.jpg")
                        leye.save(leye_path)
                    elif coord['class'] in [15, 17, 19, 21]:
                        reye = reye.crop((xmin, ymin, xmax, ymax,))
                        print('reye:xmin:{},xmax:{},ymin:{},ymax:{}'.format(xmin, xmax, ymin, ymax))
                        center_reye_roi_x, center_reye_roi_y = ((xmin + xmax) / 2), ((ymin + ymax) / 2)
                        print("x,y of  center is:", center_reye_roi_x, center_reye_roi_y)
                        reye_path = os.path.join(tmp_dir, "reye.jpg")
                        reye.save(reye_path)
                    elif coord['class'] in [22, 23, 24, 25, 26]:
                        nose = nose.crop((xmin, ymin, xmax, ymax,))
                        print('nose:xmin:{},xmax:{},ymin:{},ymax:{}'.format(xmin,xmax,ymin,ymax))
                        center_nose_roi_x,center_nose_roi_y=((xmin+xmax)/2),((ymin+ymax)/2)
                        print("x,y of nose center is:",center_nose_roi_x,center_nose_roi_y)
                        nose_path = os.path.join(tmp_dir, "nose.jpg")
                        nose.save(nose_path)
        orientation = lion_orientation_fun(face_path)
        if orientation == 'front':
            slope =0
            theta =0
        else:
            if orientation =='right':
                slope = (center_nose_roi_y - center_reye_roi_y) / (center_nose_roi_x - center_reye_roi_x)
                theta = math.degrees(math.atan(slope))-40

            else:
                slope = (center_nose_roi_y - center_leye_roi_y) / (center_nose_roi_x - center_leye_roi_x)
                theta = (math.degrees(math.atan(slope)))+40


        lion_path = os.path.join(tmp_dir, "lion.jpg")
        cv2.imwrite(lion_path, temp_image)
    return lion_path, face_path,theta


def resize(img):
    new_img = Image.open(img)
    resized_img = new_img.resize((225, 225), Image.ANTIALIAS, quality=75)
    return resized_img


def predict_not_a_lion(filename):
    from keras.preprocessing.image import load_img
    image = load_img(filename, target_size=(224, 224))
    img = np.array(image)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    model = tf.keras.models.load_model('models/vgg16.h5')
    output = model.predict(img)
    if output < 0.4:
        return 0
    else:
        return 1


def check_upload(lion_image_path):
    print("Processing image :",os.path.basename(lion_image_path))
    tmp_dir = None
    try:
        image_base_name = os.path.basename(lion_image_path)
        tmp_dir = tempfile.mkdtemp()
        pil_img = Image.open(lion_image_path)
        src = cv2.imread(lion_image_path)
        temp_image = src.copy()
        coordinates, whisker_cords, face_cords, status = lion_model.get_coordinates(lion_image_path, 'temp_lion')
        if status != "Success":
            ret = dict()
            ret['ref_face'] = get_base64_str(lion_image_path)
            ret['ref_status'] = status
            if status == "No lions detected" :
                ret['type'] = 'Not'
            logging.info("check_upload status - {0}".format(str(status)))
            return ret
        lion_path, face_path, whisker_path, lear_path, rear_path, leye_path, reye_path, nose_path, face_embedding, whisker_embedding = \
            extract_lion_data(face_cords, 'temp_lion', pil_img, coordinates, tmp_dir, temp_image)
        if whisker_path == '':
            left_whisker_path=''
            right_whisker_path=''
            whisker_spot_path=''
            left_whisker_area=0
            right_whisker_area=0
            pass
        else:
            whisker_image, whisker_spot_bb, left_whisker_path, right_whisker_path,left_whisker_area, right_whisker_area = predict_whisker(whisker_path, tmp_dir)
            whisker_spot_path = whisker_spot_calculation(whisker_spot_bb, whisker_image, tmp_dir)



        ratio_whisker,face_orientation = ratio_calculate(left_whisker_area,right_whisker_area)

        ret = dict()
        ret['ref_face'] = get_base64_str(face_path)
        ret['ref_whisker'] = get_base64_str(whisker_path)
        ret['ref_left_whisker'] = get_base64_str(left_whisker_path)
        ret['ref_right_whisker'] = get_base64_str(right_whisker_path)
        ret['ref_whisker_spot'] = get_base64_str(whisker_spot_path)
        ret['ref_lear'] = get_base64_str(lear_path)
        ret['ref_rear'] = get_base64_str(rear_path)
        ret['ref_leye'] = get_base64_str(leye_path)
        ret['ref_reye'] = get_base64_str(reye_path)
        ret['ref_nose'] = get_base64_str(nose_path)
        if (len(ret['ref_face']) == 0) or (predict_not_a_lion(face_path)):
        #if len(ret['ref_face']) == 0:
            ret['type'] = 'Not'
        else:
            match_lion(face_embedding, whisker_embedding, ret,face_orientation)
        return ret
    except Exception as e:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        return str(e)


def dd2dms(dd):
    mnt, sec = divmod(dd * 3600, 60)
    deg, mnt = divmod(mnt, 60)
    return deg, mnt, sec


def get_click_datetime(data):
    mm, dd, yyyy = data['Date'].split('/')
    time = data['UTC-Time'].split('.')[0]
    hours, mins, secs = time.split(':')
    if len(hours) < 2:
        hours = '0' + hours
    if len(mins) < 2:
        mins = '0' + mins
    if len(secs) < 2:
        secs = '0' + secs
    new_time = hours + ':' + mins + ':' + secs
    iso_dt = yyyy + '-' + mm + '-' + dd + ' ' + new_time
    dt = datetime.fromisoformat(iso_dt)
    return dt


def upload_one_lion(lion_image_path, lion_name,microchip_number, gender, l_status,l_age,microchip_number_new='',latitude='',longitude =''):
    image_name=os.path.basename(lion_image_path)
    tmp_dir = tempfile.mkdtemp()
    lion_gender = gender
    lion_status = l_status
    lion_age = l_age
    microchip_number_new =microchip_number_new
    face_orientation='U' #u=Uknown
    try:
        lat = latitude
        lon = longitude
        utc_click_datetime = datetime.now(timezone.utc)
        lion_id = str(current_milli_time())
        #data = gpsphoto.getGPSData(lion_image_path)
        data = {}
        hash_value = img_hash_value(lion_image_path)
        # duplicated image detection
        str_hash_value = str(hash_value)
        dup_val, status = duplicate_img_detected(str_hash_value)
        if dup_val == 1:
            r = dict()
            r['lion_name'] = lion_name
            r['microchip_number'] = microchip_number
            r['lion_image_file_name'] = os.path.join(lion_image_path)
            r['status'] = status
            r['microchip_number_new'] = microchip_number_new
            return r
        else:
            if len(data) > 0:
                try:
                    lat_deg, lat_mnt, lat_sec = dd2dms(data['Latitude'])
                    lat = f"{lat_deg}° {lat_mnt}' {lat_sec}\""
                except Exception as e:
                    lat = f"{0.0}° {0.0}' {0.0}\""
                try:
                    lon_deg, lon_mnt, lon_sec = dd2dms(data['Longitude'])
                    lon = f"{lon_deg}° {lon_mnt}' {lon_sec}\""
                except Exception as e:
                    lon = f"{0.0}° {0.0}' {0.0}\""
                try:
                    utc_click_datetime = get_click_datetime(data)
                except Exception as e:
                    utc_click_datetime = datetime.now(timezone.utc)

        pil_img = Image.open(lion_image_path)
        src = cv2.imread(lion_image_path)
        temp_image = src.copy()
        coordinates, whisker_cords, face_cords, status = lion_model.get_coordinates(lion_image_path, lion_name)

        if status != "Success":
            logging.info("upload_one_lion - {0}".format(str(status)))
            r = dict()
            r['lion_name'] = lion_name
            r['microchip_number'] = microchip_number
            r['lion_image_file_name'] = os.path.basename(lion_image_path)
            r['status'] = status
            r['microchip_number_new'] =microchip_number_new
            return r

        # for compressed_data
        resize_temp_image = cv2.resize(src, (30, 30), interpolation=cv2.INTER_NEAREST)
        c_lion_path, c_face_path, c_whisker_path, c_lear_path, c_rear_path, c_leye_path, c_reye_path, c_nose_path, c_face_embedding, c_whisker_embedding = \
                extract_lion_data(face_cords, lion_name, pil_img, coordinates, tmp_dir, resize_temp_image)

        insert_compressed_data(lion_id,microchip_number, lion_name, c_lion_path,
                                   c_face_path, c_whisker_path,
                                   c_lear_path, c_rear_path,
                                   c_leye_path, c_reye_path,
                                   c_nose_path,microchip_number_new)


        lion_path, face_path, whisker_path, lear_path,rear_path,leye_path,reye_path, nose_path, face_embedding, whisker_embedding =\
            extract_lion_data(face_cords, lion_name, pil_img, coordinates, tmp_dir, temp_image)
        if whisker_path == '':
            left_whisker_area=0
            right_whisker_area=0
            left_whisker_path=''
            right_whisker_path=''
            whisker_spot_path=''
            pass

        else:

            whisker_image, whisker_spot_bb, left_whisker_path, right_whisker_path,left_whisker_area,right_whisker_area = predict_whisker(whisker_path, tmp_dir)
            whisker_spot_path = whisker_spot_calculation(whisker_spot_bb, whisker_image, tmp_dir)

        ratio_whisker,face_orientation=ratio_calculate(left_whisker_area,right_whisker_area)

        print("ratio and face orientation in upload one lion",ratio_whisker,face_orientation)

        insert_lion_data(lion_id,image_name, microchip_number,lion_name,
                         lion_gender, lion_status,
                         utc_click_datetime,
                         lat, lon, lion_path,
                         face_path, whisker_path,
                         left_whisker_path,right_whisker_path,
                         whisker_spot_path,
                         lear_path, rear_path,
                         leye_path, reye_path,
                         nose_path, face_embedding,
                         whisker_embedding,hash_value,lion_age,ratio_whisker,face_orientation,False,microchip_number_new)

        # face_bytes = get_base64_str(face_path)
        shutil.rmtree(tmp_dir)
        r = dict()
        r['lion_name'] = lion_name
        r['lion_image_file_name'] = os.path.basename(lion_image_path)
        # r['image'] = face_bytes
        r['status'] = 'Success'
        r['microchip_number_new'] = microchip_number_new
        return r

    except Exception as e:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
            r = dict()
            r['lion_name'] = lion_name
            r['microchip_number'] = microchip_number
            r['lion_image_file_name'] = os.path.basename(lion_image_path)
            r['status'] = str(e)
            r['microchip_number_new'] = microchip_number_new
            return r


def on_board_new_lion(lion,microchip_number, lion_dir, rv, second, job_id,processed_images):

    # dataframe for setting flag =True when frontal face lion image
    whisker_ratio_list=[]
    # id_df = []
    # microchip_number_df = []
    # ratio_df = []

    tmp_dir = None
    lion_images = os.listdir(lion_dir)
    individual_lion_images = 0
    for lion_image in lion_images:
        face_orientation='U'

        individual_lion_images += 1
        total_processed_image = processed_images + individual_lion_images
        update_process_tracker(job_id=job_id, file_processed=total_processed_image, current_status="in-progress")

        try:
            lat = ""
            lon = ""
            utc_click_datetime = datetime.now(timezone.utc)
            lion_id = str(current_milli_time())
            tmp_dir = tempfile.mkdtemp()
            lion_image_path = os.path.join(lion_dir, lion_image)
            try:
                data = gpsphoto.getGPSData(lion_image_path)
            except Exception as e:
                data = {}
            if len(data) > 0:
                try:
                    lat_deg, lat_mnt, lat_sec = dd2dms(data['Latitude'])
                    lat = f"{lat_deg}° {lat_mnt}' {lat_sec}\""
                except Exception as e:
                    lat = f"{0.0}° {0.0}' {0.0}\""
                try:
                    lon_deg, lon_mnt, lon_sec = dd2dms(data['Longitude'])
                    lon = f"{lon_deg}° {lon_mnt}' {lon_sec}\""
                except Exception as e:
                    lon = f"{0.0}° {0.0}' {0.0}\""
                try:
                    utc_click_datetime = get_click_datetime(data)
                except Exception as e:
                    utc_click_datetime = datetime.now(timezone.utc)
            pil_img = Image.open(lion_image_path)
            src = cv2.imread(lion_image_path)
            temp_image = src.copy()
            hash_value = img_hash_value(lion_image_path)
            #print(hash_value)
            # duplicated image detection
            str_hash_value = str(hash_value)
            dup_val, status = duplicate_img_detected(str_hash_value)
            if dup_val == 1:
                logging.info("on_board_new_lion - {0}".format(str(status)))
                r = dict()
                r['lion_name'] = lion
                r['microchip_number'] = microchip_number
                r['lion_image_file_name'] = lion_image
                r['status'] = status
                rv['status'].append(r)
                continue

            # for compressed_data
            resize_temp_image = cv2.resize(src, (30, 30), interpolation=cv2.INTER_NEAREST)
            coordinates, whisker_cords, face_cords, status = lion_model.get_coordinates(lion_image_path, lion)
            if face_cords != None:
                c_lion_path, c_face_path, c_whisker_path, c_lear_path, c_rear_path, c_leye_path, c_reye_path, c_nose_path, c_face_embedding, c_whisker_embedding = \
                    extract_lion_data(face_cords, lion, pil_img, coordinates, tmp_dir, resize_temp_image)
            else:
                status = "Face not detected"

            # insert_compressed_data(lion_id, lion, c_lion_path,
            #                        c_face_path, c_whisker_path,
            #                        c_lear_path, c_rear_path,
            #                        c_leye_path, c_reye_path,
            #                        c_nose_path)


            if status != "Success":
                logging.info("on_board_new_lion - {0}".format(str(status)))
                r = dict()
                r['lion_name'] = lion
                r['microchip_number'] = microchip_number
                r['lion_image_file_name'] = lion_image
                r['status'] = status
                rv['status'].append(r)
                continue
            lion_path, face_path, whisker_path, lear_path, rear_path, leye_path, reye_path, nose_path, face_embedding, whisker_embedding = \
                extract_lion_data(face_cords, lion, pil_img, coordinates, tmp_dir, temp_image)

            #left_whisker , right whisker , whisker spot calculation
            if whisker_path == '':
                left_whisker_path=''
                right_whisker_path=''
                whisker_spot_path=''
                left_whisker_area=0
                right_whisker_area=0
                pass
            else:

                whisker_image, whisker_spot_bb, left_whisker_path, right_whisker_path,left_whisker_area,right_whisker_area = predict_whisker( whisker_path, tmp_dir)
                whisker_spot_path = whisker_spot_calculation(whisker_spot_bb, whisker_image, tmp_dir)

            ratio_whisker,face_orientation=ratio_calculate(left_whisker_area,right_whisker_area)

            if len(whisker_embedding) > 0 and len(face_embedding) > 0:
                ret = dict()
                if second:
                    match_lion(face_embedding, whisker_embedding, ret,face_orientation)
                    if ret['type'] == 'Not':
                        r = dict()
                        r['lion_name'] = lion
                        r['lion_image_file_name'] = lion_image
                        r['status'] = 'Not a lion'
                        rv['status'].append(r)
                    else:

                        insert_lion_data(lion_id,lion_image,microchip_number,lion,
                                         'U', 'A',
                                         utc_click_datetime,
                                         lat, lon, lion_path,
                                         face_path, whisker_path,
                                         left_whisker_path,right_whisker_path,
                                         whisker_spot_path,
                                         lear_path, rear_path,
                                         leye_path, reye_path,
                                         nose_path, face_embedding,
                                         whisker_embedding, hash_value,0,ratio_whisker,face_orientation,False,'')

                        insert_compressed_data(lion_id,microchip_number, lion, c_lion_path,
                                               c_face_path, c_whisker_path,
                                               c_lear_path, c_rear_path,
                                               c_leye_path, c_reye_path,
                                               c_nose_path)
                        r = dict()
                        r['lion_name'] = lion
                        r['microchip_number'] = microchip_number
                        r['lion_image_file_name'] = lion_image
                        r['status'] = 'Success'
                        rv['status'].append(r)
                else:
                    insert_lion_data(lion_id,lion_image, microchip_number, lion,
                                     'U', 'A',
                                     utc_click_datetime,
                                     lat, lon, lion_path,
                                     face_path, whisker_path,
                                     left_whisker_path, right_whisker_path, whisker_spot_path,
                                     lear_path, rear_path,
                                     leye_path, reye_path,
                                     nose_path, face_embedding,
                                     whisker_embedding, hash_value, 0, ratio_whisker,face_orientation,False,'')
                    r = dict()
                    r['lion_name'] = lion
                    r['microchip_number'] = microchip_number
                    r['lion_image_file_name'] = lion_image
                    r['status'] = 'Success'
                    rv['status'].append(r)
                    insert_compressed_data(lion_id,microchip_number, lion, c_lion_path,
                                           c_face_path, c_whisker_path,
                                           c_lear_path, c_rear_path,
                                           c_leye_path, c_reye_path,
                                           c_nose_path)
            else:
                r = dict()
                r['lion_name'] = lion
                r['microchip_number'] = microchip_number
                r['lion_image_file_name'] = lion_image
                r['status'] = 'Either face or whisker is not detected (may be not a lion or unclear image)'
                rv['status'].append(r)
                shutil.rmtree(tmp_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            r = dict()
            r['lion_name'] = lion
            r['microchip_number'] = microchip_number
            r['lion_image_file_name'] = lion_image
            r['status'] = str(e)
            rv['status'].append(r)
            update_process_tracker(job_id=job_id, file_processed=total_processed_image, current_status="exception")

        # id_df.append(lion_id)
        # microchip_number_df.append(microchip_number)
        # ratio_df.append(ratio_whisker)
        whisker_ratio_list.append([lion_id,microchip_number,ratio_whisker,face_orientation])

    # whisker_ratio_list = sorted(whisker_ratio_list, lambda x: x[2])
    # whisker_ratio_list = sorted(whisker_ratio_list, lambda x: x[2])
    whisker_ratio_list = sorted(whisker_ratio_list, key=lambda x: x[2])

    print("whisker_ratio_list:",whisker_ratio_list)

    # import pandas as pd
    # df = pd.DataFrame()
    # df['id_lion'] = id_df
    # df['microchip_no'] = microchip_number_df
    # df['ratio'] = ratio_df
    # print(df)

    for each_list in whisker_ratio_list:
        lion_id_updated=each_list[0]
        face_orientation_updated=each_list[3]
    update_lion_data(lion_id=lion_id_updated,face_orientation=face_orientation_updated)
    try:
        #SETTING FLAG=TRUE for minimum ratio value in paricular face orientation (say F)
        minimum_ratio_list = []
        result_F = list(filter(lambda x: x[3] == 'F', whisker_ratio_list))
        result_L = list(filter(lambda x: x[3] == 'L', whisker_ratio_list))
        result_R = list(filter(lambda x: x[3] == 'R', whisker_ratio_list))

        if len(result_F) >0:
            minimum_ratio_list.append(result_F[0][0])
        if len(result_L) > 0:
            minimum_ratio_list.append(result_L[0][0])
        if len (result_R) > 0:
            minimum_ratio_list.append(result_R[0][0])
        print("minimum_ratio_list:",minimum_ratio_list)
        for each in minimum_ratio_list:
            update_lion_data_flag(lion_id=each)
    except Exception as e:
        import traceback
        traceback.print_exc()

    update_process_tracker(job_id=job_id, file_processed=total_processed_image, current_status="completed")