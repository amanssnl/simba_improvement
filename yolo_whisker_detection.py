import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import imutils

cwd = os.getcwd()

def predict_whisker_yolo(face,tmp_dir,theta):
  try:
    src = cv2.imread(face)
    temp_image = src.copy()
    rotated_image = imutils.rotate(src, theta)
    path_rotated = os.path.join(tmp_dir, 'rotated.jpg')
    print(path_rotated)
    cv2.imwrite(path_rotated, rotated_image)
    # window_name = 'Rotate Image by Angle in Python'
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.imshow(window_name, rotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    pil_img = Image.open(path_rotated)
    cwd = os.getcwd()

    model_path = 'models/yolov4-custom_best.weights'
    conf_path = 'models/yolov4-custom.cfg'
    yolo = cv2.dnn.readNet(model_path, conf_path)
    print("yolo Model init done")
    classes = []
    with open("models/whisker_spot_class.txt", 'r') as f:
      classes = f.read().splitlines()

    img=cv2.imread(path_rotated)
    height, width, channels = img.shape
    blob=cv2.dnn.blobFromImage(img,1/255,(320,320),(0,0,0),swapRB=True,crop=False)
    yolo.setInput(blob)
    output_layers_names= yolo.getUnconnectedOutLayersNames()
    layeroutput= yolo.forward(output_layers_names)

    boxes=[]
    confidences =[]
    class_ids=[]
    for output in layeroutput:

      for detection in output:
        score=detection[5:]
        class_id=np.argmax(score)
        confidence=score[class_id]

        if confidence > 0.7:
          center_x=int(detection[0]*width)
          center_y= int(detection[1]*height)
          w= int(detection[2]*width)
          h= int(detection[3]*height)
          x=int(center_x-w/2)
          y=int(center_y-h/2)
          boxes.append([x,y,w,h])
          confidences.append(float(confidence))
          class_ids.append(class_id)

    indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    font =cv2.FONT_HERSHEY_PLAIN
    colors=np.random.uniform(0,255,size=(len(boxes),3))
    center_bounding_box = []
    # print(boxes)
    whisker_path= ''

    for i in range(len(boxes)):
      if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        # print(label)

        try:
          if label == 'whisker':
            whisker = img[y:y+ h, x:x + w]
            whisker_path = os.path.join(tmp_dir, "whisker.jpg")
            cv2.imwrite(whisker_path, whisker)
        except:
          whisker_path=''

    #     color = colors[class_ids[i]]
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
    #     cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    #                 1 / 2, color, 2)
    #
    #
    # cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)  # Create window with freedom of dimensions
    # imS = cv2.resize(img, (960, 540))  # Resize image
    # cv2.imshow("output", imS)  # Show image
    # cv2.waitKey(0)
    print('whisker_path yolo line 101', whisker_path)
    return whisker_path

  except Exception as e :
    import traceback
    traceback.print_exc()
    return whisker_path

# rotated_image,whiser_path= predict_whisker_yolo('input_dir/face.jpg','tmp_dir',-40+50)