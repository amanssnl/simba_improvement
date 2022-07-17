import os
import numpy as np
from PIL import Image
from collections import defaultdict

from keras.applications.resnet50 import preprocess_input

path_base = 'data'
path_csv = 'face.csv'
embedding_dim = 50
image_size = 224
batch_size = 48
path_train = os.path.join(path_base, 'face')


class SampleGen(object):
    def __init__(self, file_class_mapping, other_class="new_lion"):
        self.file_class_mapping = file_class_mapping
        self.class_to_list_files = defaultdict(list)
        self.list_other_class = []
        self.list_all_files = list(file_class_mapping.keys())
        self.range_all_files = list(range(len(self.list_all_files)))

        for file, class_ in file_class_mapping.items():
            if class_ == other_class:
                self.list_other_class.append(file)
            else:
                self.class_to_list_files[class_].append(file)

        self.list_classes = list(set(self.file_class_mapping.values()))
        self.range_list_classes = range(len(self.list_classes))
        self.class_weight = np.array([len(self.class_to_list_files[class_]) for class_ in self.list_classes])
        self.class_weight = self.class_weight / np.sum(self.class_weight)

    def get_sample(self):
        class_idx = np.random.choice(self.range_list_classes, 1, p=self.class_weight)[0]

        v1 = self.list_classes[class_idx]
        v2 = self.class_to_list_files[v1]
        v3 = len(v2)
        v4 = range(v3)
        examples_class_idx = np.random.choice(v4, 2)

        y1 = examples_class_idx[0]
        y2 = examples_class_idx[1]

        positive_example_1, positive_example_2 = v2[y1], v2[y2]

        negative_example = None
        negative_example_class = None
        positive_example_1_class = None

        if negative_example:
            negative_example_class = self.file_class_mapping[negative_example]
            positive_example_1_class = self.file_class_mapping[positive_example_1]

        while negative_example is None or negative_example_class == positive_example_1_class:
            negative_example_idx = np.random.choice(self.range_all_files, 1)[0]
            negative_example = self.list_all_files[negative_example_idx]
            if negative_example:
                negative_example_class = self.file_class_mapping[negative_example]
                positive_example_1_class = self.file_class_mapping[positive_example_1]

        return positive_example_1, negative_example, positive_example_2


def read_and_resize(filepath):
    im = Image.open(filepath).convert('RGB')
    im = im.resize((image_size, image_size))
    return np.array(im, dtype="float32")


def augment(im_array):
    if np.random.uniform(0, 1) > 0.9:
        im_array = np.fliplr(im_array)
    return im_array


def gen(triplet_gen):
    while True:
        list_positive_examples_1 = []
        list_negative_examples = []
        list_positive_examples_2 = []

        for i in range(batch_size):
            positive_example_1, negative_example, positive_example_2 = triplet_gen.get_sample()
            path_pos1 = os.path.join(path_train, positive_example_1)
            path_neg = os.path.join(path_train, negative_example)
            path_pos2 = os.path.join(path_train, positive_example_2)

            positive_example_1_img = read_and_resize(path_pos1)
            negative_example_img = read_and_resize(path_neg)
            positive_example_2_img = read_and_resize(path_pos2)

            positive_example_1_img = augment(positive_example_1_img)
            negative_example_img = augment(negative_example_img)
            positive_example_2_img = augment(positive_example_2_img)

            list_positive_examples_1.append(positive_example_1_img)
            list_negative_examples.append(negative_example_img)
            list_positive_examples_2.append(positive_example_2_img)

        a = preprocess_input(np.array(list_positive_examples_1))
        b = preprocess_input(np.array(list_positive_examples_2))
        c = preprocess_input(np.array(list_negative_examples))

        label = None

        yield {'anchor_input': a, 'positive_input': b, 'negative_input': c}, label
