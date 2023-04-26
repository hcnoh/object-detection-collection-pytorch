import os
import pickle

import xml.etree.ElementTree as Et

from PIL import Image
from torch.utils.data import Dataset
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


DATASET_DIR = "datasets"

VOC2012_TRAINVAL_DIR = os.path.join(DATASET_DIR, "voc2012-trainval")
VOC2007_TRAINVAL_DIR = os.path.join(DATASET_DIR, "voc2007-trainval")
VOC2007_TEST_DIR = os.path.join(DATASET_DIR, "voc2007-test")

CLASS_LIST = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class VOC:
    def __init__(
        self,
        dataset_dir=DATASET_DIR,
    ) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir

        self.preprocessed_dataset_path = os.path.join(
            self.dataset_dir, "voc_dataset.pkl"
        )

        self.cls_list = CLASS_LIST
        self.cls2idx = {c: i for i, c in enumerate(self.cls_list)}

        if os.path.exists(self.preprocessed_dataset_path):
            with open(self.preprocessed_dataset_path, "rb") as f:
                (
                    self.train_img_path_list,
                    self.train_lbl_list,
                    self.val_img_path_list,
                    self.val_lbl_list,
                ) = pickle.load(f)

        else:
            self.preprocess()

        self.train_dataset = VOCDataset(
            self.train_img_path_list, self.train_lbl_list
        )
        self.val_dataset = VOCDataset(
            self.val_img_path_list, self.val_lbl_list
        )

    def get_path_lists(self, root_annotation_path_list):
        img_path_list = []
        lbl_path_list = []

        for root_path in root_annotation_path_list:
            for dir_path, _, file_names in os.walk(root_path):
                for file_name in file_names:
                    img_path_list.append(
                        os.path.join(
                            dir_path.replace("Annotations", "JPEGImages"),
                            file_name.replace(".xml", ".jpg"),
                        )
                    )
                    lbl_path_list.append(
                        os.path.join(
                            dir_path,
                            file_name,
                        )
                    )

        return img_path_list, lbl_path_list

    def get_lbl_list(self, lbl_path_list):
        lbl_list = []

        for lbl_path in lbl_path_list:
            lbl = []

            with open(lbl_path, "r") as xml:
                tree = Et.parse(xml)
                root = tree.getroot()

                objects = root.findall("object")

                for obj in objects:
                    name = obj.find("name").text
                    bbox = obj.find("bndbox")

                    x1 = int(float(bbox.find("xmin").text))
                    x2 = int(float(bbox.find("xmax").text))
                    y1 = int(float(bbox.find("ymin").text))
                    y2 = int(float(bbox.find("ymax").text))

                    lbl.append(
                        BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2, label=name)
                    )

                lbl_list.append(lbl)

        return lbl_list

    def preprocess(self):
        train_img_path_list, train_lbl_path_list = self.get_path_lists(
            [
                os.path.join(
                    VOC2012_TRAINVAL_DIR,
                    "VOCdevkit",
                    "VOC2012",
                    "Annotations",
                ),
                os.path.join(
                    VOC2007_TRAINVAL_DIR,
                    "VOCdevkit",
                    "VOC2007",
                    "Annotations",
                ),
            ]
        )

        train_lbl_list = self.get_lbl_list(
            train_lbl_path_list,
        )

        val_img_path_list, val_lbl_path_list = self.get_path_lists(
            [
                os.path.join(
                    VOC2007_TEST_DIR,
                    "VOCdevkit",
                    "VOC2007",
                    "Annotations",
                ),
            ]
        )

        val_lbl_list = self.get_lbl_list(
            val_lbl_path_list,
        )

        self.train_img_path_list = train_img_path_list
        self.train_lbl_list = train_lbl_list
        self.val_img_path_list = val_img_path_list
        self.val_lbl_list = val_lbl_list

        with open(self.preprocessed_dataset_path, "wb") as f:
            pickle.dump(
                (
                    self.train_img_path_list,
                    self.train_lbl_list,
                    self.val_img_path_list,
                    self.val_lbl_list,
                ),
                f
            )


class VOCDataset(Dataset):
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
