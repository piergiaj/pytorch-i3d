import os
import csv

from collections import namedtuple

ListDataJpeg = namedtuple('ListDataJpeg', ['id', 'label', 'action', 'scene', 'path'])

class JpegDataset(object):

    def __init__(self, csv_path_input, csv_path_action_labels, csv_path_scene_labels, data_root):
        self.csv_data = self.read_csv_input(csv_path_input, data_root)
        self.action_classes = self.read_csv_labels(csv_path_action_labels)
        self.scene_classes = self.read_csv_labels(csv_path_scene_labels)
        self.action_classes_dict = self.get_two_way_dict(self.action_classes)
        self.scene_classes_dict = self.get_two_way_dict(self.scene_classes)

    def read_csv_input(self, csv_path, data_root):
        csv_data = []
        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')
            for row in csv_reader:
                item = ListDataJpeg(row[0],
                                    row[1],
                                    row[2], # action
                                    row[3], # scene
                                    os.path.join(data_root, row[0])
                                    )
                csv_data.append(item)
        return csv_data

    def read_csv_labels(self, csv_path):
        classes = []
        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                classes.append(row[0])
        return classes

    def get_two_way_dict(self, classes):
        classes_dict = {}
        for i, item in enumerate(classes):
            classes_dict[item] = i
            classes_dict[i] = item
        return classes_dict

    # To do
    def summarize(self):
        # number of instances
        print("\nNumber of instances ----------------------------------------")
        print(self.data_df.shape[0])

        # number of classes
        print("\nNumber of classes ------------------------------------------")
        print(self.data_df.template.nunique())

        # number of instances per class
        print("\nNumber of instances per class ------------------------------")
        print(self.data_df.template.value_counts())

        # max and min duration of vides
        print("\nMax, Min, Avg, Std duration --------------------------------")
        print(self.data_df.duration.max())
        print(self.data_df.duration.min())
        print(self.data_df.duration.mean())
        print(self.data_df.duration.std())

        print("\n")

