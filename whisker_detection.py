

import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import numpy as np
from PIL import Image
import os
import imutils

CLASS_NAMES = ['BG', 'whisker']


class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = len(CLASS_NAMES)


model = mrcnn.model.MaskRCNN(mode="inference",
                             config=SimpleConfig(),
                             model_dir='models/mask_rcnn_maskrcnn_config_0009.h5')

model.load_weights(filepath='models/mask_rcnn_maskrcnn_config_0009.h5',
                   by_name=True)

# image = cv2.imread('input_dir/left_6S2A8888_face.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# r = model.detect([image], verbose=0)
# r=r[0]


def whisker_roi(face,tmp_dir,theta):
    try:
        whisker_path =''
        # pil_img = Image.open(face)
        # face = pil_img.copy()
        src = cv2.imread(face)
        temp_image = src.copy()

        # we are using
        rotate_image = imutils.rotate(src, theta)
        path_cropped=os.path.join(tmp_dir,'cropped.jpg')
        cv2.imwrite(path_cropped, rotate_image)
        pil_img = Image.open(path_cropped)
        face = pil_img.copy()
        # window_name = 'Rotate Image by Angle in Python'
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(window_name, rotate_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        image = cv2.cvtColor(rotate_image, cv2.COLOR_BGR2RGB)
        r = model.detect([image], verbose=0)
        r = r[0]
        print("R:",r)
        if (np.max(r['scores']) > 0.80):
            # mrcnn.visualize.display_instances(image=image,
            #                                   boxes=r['rois'],
            #                                   masks=r['masks'],
            #                                   class_ids=r['class_ids'],
            #                                   class_names=CLASS_NAMES,
            #                                   scores=r['scores'])

            xmin = r['rois'][0][1]
            xmax = r['rois'][0][3]
            ymin = r['rois'][0][0]
            ymax = r['rois'][0][2]

        temp_image = cv2.rectangle(temp_image,
                                   (xmin, ymin),
                                   (xmax, ymax),
                                   (36, 255, 12),
                                   4)

        whisker = pil_img.crop((xmin, ymin, xmax, ymax,))
        whisker_path = os.path.join(tmp_dir, "whisker.jpg")
        whisker.save(whisker_path)

    except Exception as e:
        whisker_path=''

    return whisker_path

# print(whisker_roi('input_dir/face.jpg','tmp_dir'))
