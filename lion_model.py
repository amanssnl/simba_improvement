import os
from lion_detector import LionDetector


classes = {
    "1": "cv-dl",
    "2": "cv-dr",
    "3": "cv-f",
    "4": "cv-sl",
    "5": "cv-sr",
    "6": "ear-dl-l",
    "7": "ear-dl-r",
    "8": "ear-dr-l",
    "9": "ear-dr-r",
    "10": "ear-fl",
    "11": "ear-fr",
    "12": "ear-sl",
    "13": "ear-sr",
    "14": "eye-dl-l",
    "15": "eye-dl-r",
    "16": "eye-dr-l",
    "17": "eye-dr-r",
    "18": "eye-fl",
    "19": "eye-fr",
    "20": "eye-sl",
    "21": "eye-sr",
    "22": "nose-dl",
    "23": "nose-dr",
    "24": "nose-f",
    "25": "nose-sl",
    "26": "nose-sr",
    "27": "whisker-dl",
    "28": "whisker-dr",
    "29": "whisker-f",
    "30": "whisker-sl",
    "31": "whisker-sr",
}


class LionDetection:

    def __init__(self):
        cwd = os.getcwd()
        model_path = os.path.join(cwd, 'models', 'lion_detection_model.pth')
        self.model = LionDetector(model_path)

    def get_coordinates(self, image_path, name):
        image_whiskers = dict()
        image_faces = dict()
        try:
            results, time_taken = self.model.detect(image_path, name, 0.90)
        except Exception as e:
            return None, None, None, str(e)
        time_taken += time_taken
        whiskers = []
        face_coordinates = []
        for box in results['boxes']:
            roi_class = classes[str(box['class'])]
            if box['class'] in [27, 28, 29, 30, 31]:
                whiskers.append(box)
            if box['class'] in [1, 2, 3, 4, 5]:
                face_coordinates.append(box)
        if len(whiskers) == 0 and len(face_coordinates) == 0:
            return None, None, None, "No lions detected"
        if len(whiskers) != 1 and len(face_coordinates) != 1:
            return None, None, None, "Multiple lions detected"
        image_whiskers[name] = {"name": name, "boxes": whiskers}
        image_faces[name] = {"name": name, "boxes": face_coordinates}
        return results, image_whiskers, image_faces, "Success"

    def insideface(self, face_coord, parts_coord):
        face_coord_bbox = face_coord['ROI']
        parts_coord_bbox = parts_coord['ROI']
        x_left = max(face_coord_bbox[0], parts_coord_bbox[0])
        y_top = max(face_coord_bbox[1], parts_coord_bbox[1])
        x_right = min(face_coord_bbox[2], parts_coord_bbox[2])
        y_bottom = min(face_coord_bbox[3], parts_coord_bbox[3])

        if x_right < x_left or y_bottom < y_top:
            return False

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (face_coord_bbox[2] - face_coord_bbox[0]) * (face_coord_bbox[3] - face_coord_bbox[1])
        bb2_area = (parts_coord_bbox[2] - parts_coord_bbox[0]) * (parts_coord_bbox[3] - parts_coord_bbox[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return True
