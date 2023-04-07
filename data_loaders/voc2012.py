import os
import pickle

import numpy as np
import xml.etree.ElementTree as Et

from PIL import Image
# from xml.etree.ElementTree import Element, ElementTree
from torch.utils.data import Dataset
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


DATASET_DIR = "datasets/voc2012"


class VOC2012:
    def __init__(
        self,
        dataset_dir=DATASET_DIR,
    ) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir

        self.preprocessed_dataset_path = os.path.join(
            self.dataset_dir, "dataset.pkl"
        )

        self.imgs_path = os.path.join(
            self.dataset_dir,
            "VOCdevkit",
            "VOC2012",
            "JPEGImages",
        )
        self.lbls_path = os.path.join(
            self.dataset_dir,
            "VOCdevkit",
            "VOC2012",
            "Annotations",
        )

        self.train_list_path = os.path.join(
            self.dataset_dir,
            "VOCdevkit",
            "VOC2012",
            "ImageSets",
            "Main",
            "train.txt",
        )

        self.val_list_path = os.path.join(
            self.dataset_dir,
            "VOCdevkit",
            "VOC2012",
            "ImageSets",
            "Main",
            "val.txt",
        )

        if os.path.exists(self.preprocessed_dataset_path):
            with open(self.preprocessed_dataset_path, "rb") as f:
                (
                    self.train_img_paths,
                    self.train_lbls,
                    self.val_img_paths,
                    self.val_lbls,
                    self.cls_list,
                    self.cls2idx,
                ) = pickle.load(f)

        else:
            self.preprocess()

        self.train_dataset = VOC2012Dataset(
            self.train_img_paths, self.train_lbls
        )
        self.val_dataset = VOC2012Dataset(
            self.val_img_paths, self.val_lbls
        )

    def preprocess(self):
        self.train_img_paths = []
        self.train_lbls = []

        self.val_img_paths = []
        self.val_lbls = []

        cls_list = []

        with open(self.train_list_path, "r") as f:
            for line in f.readlines():
                img_path, lbl = self.preprocess_one_step(line)

                self.train_img_paths.append(img_path)
                self.train_lbls.append(lbl)

                cls_list.extend([bbox.label for bbox in lbl])

        with open(self.val_list_path, "r") as f:
            for line in f.readlines():
                img_path, lbl = self.preprocess_one_step(line)

                self.val_img_paths.append(img_path)
                self.val_lbls.append(lbl)

                cls_list.extend([bbox.label for bbox in lbl])

        self.cls_list = np.unique(cls_list)
        self.cls2idx = {cls: i for i, cls in enumerate(self.cls_list)}

        with open(self.preprocessed_dataset_path, "wb") as f:
            pickle.dump(
                (
                    self.train_img_paths,
                    self.train_lbls,
                    self.val_img_paths,
                    self.val_lbls,
                    self.cls_list,
                    self.cls2idx,
                ),
                f,
            )

    def preprocess_one_step(self, line):
        line = line.strip()
        img_path = os.path.join(self.imgs_path, "{}.jpg".format(line))

        lbl = []

        with open(
            os.path.join(self.lbls_path, "{}.xml".format(line)), "r"
        ) as xml:
            tree = Et.parse(xml)
            root = tree.getroot()

            objects = root.findall("object")

            for obj in objects:
                name = obj.find("name").text
                bbox = obj.find("bndbox")

                x1 = int(bbox.find("xmin").text)
                x2 = int(bbox.find("xmax").text)
                y1 = int(bbox.find("ymin").text)
                y2 = int(bbox.find("ymax").text)

                lbl.append(BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2, label=name))

        return img_path, lbl


class VOC2012Dataset(Dataset):
    def __init__(self, img_paths, lbls) -> None:
        super().__init__()

        self.img_paths = img_paths
        self.lbls = lbls

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = Image.open(img_path)
        lbl = BoundingBoxesOnImage(self.lbls[index], shape=img.size)

        return index, img, lbl

    def __len__(self):
        return len(self.img_paths)
