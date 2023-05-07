import os
import datetime
import json
import warnings
import copy

import torch
import torch.cuda
import torch.backends.mps

from torch.utils.data import DataLoader

from config import DEVICE, BATCH_SIZE, MODEL_CONFIG, TRAIN_CONFIG
from data_loaders.voc import VOC
from models.yolov1 import YOLOv1
from models.yolov2 import YOLOv2

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    model_name = "YOLOv2"

    ckpt_path = "ckpts"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, model_name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    now = datetime.datetime.now()
    ckpt_path = os.path.join(ckpt_path, now.strftime("%Y-%m-%d %H-%M-%S"))
    os.makedirs(ckpt_path)

    dataset = VOC()

    train_dataset = dataset.train_dataset
    val_dataset = dataset.val_dataset

    model_config = MODEL_CONFIG[model_name]
    model_config["cls_list"] = dataset.cls_list
    model_config["cls2idx"] = dataset.cls2idx

    train_config = TRAIN_CONFIG[model_name]["VOC2012"]
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        train_config_to_restore = copy.deepcopy(train_config)
        train_config_to_restore["batch_size"] = BATCH_SIZE

        json.dump(train_config_to_restore, f, indent=4)

    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("error")
    elif DEVICE == "mps" and not torch.backends.mps.is_available():
        print("error")

    print(DEVICE)

    if model_name == "YOLOv1":
        model = YOLOv1(**model_config).to(DEVICE)
    elif model_name == "YOLOv2":
        model = YOLOv2(**model_config).to(DEVICE)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=model.collate_fn_with_imgaug
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=True,
        collate_fn=model.collate_fn
    )

    train_config["train_loader"] = train_loader
    train_config["val_loader"] = val_loader
    train_config["ckpt_path"] = ckpt_path

    model.train_model(**train_config)

    # model.evaluate_model(val_loader, ckpt_path)


if __name__ == "__main__":
    main()
