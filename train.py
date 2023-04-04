import os
import datetime
import json
import warnings

import torch
import torch.cuda
import torch.backends.mps

from torch.utils.data import DataLoader

from config import DEVICE, BATCH_SIZE, MODEL_CONFIG, TRAIN_CONFIG
from data_loaders.voc2012 import VOC2012
from models.yolov1 import YOLOv1

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ckpt_path = "ckpts"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    now = datetime.datetime.now()
    ckpt_path = os.path.join(ckpt_path, now.strftime("%Y-%m-%d %H-%M-%S"))
    os.makedirs(ckpt_path)

    dataset = VOC2012()

    train_dataset = dataset.train_dataset
    val_dataset = dataset.val_dataset

    model_config = MODEL_CONFIG["YOLOv1"]
    model_config["cls_list"] = dataset.cls_list
    model_config["cls2idx"] = dataset.cls2idx

    train_config = TRAIN_CONFIG["YOLOv1"]["VOC2012"]
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f)

    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("error")
    elif DEVICE == "mps" and not torch.backends.mps.is_available():
        print("error")

    print(DEVICE)

    model = YOLOv1(**model_config).to(DEVICE)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=model.collate_fn_with_imgaug
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=model.collate_fn
    )

    train_config["train_loader"] = train_loader
    train_config["val_loader"] = val_loader
    train_config["ckpt_path"] = ckpt_path

    model.train_model(**train_config)


if __name__ == "__main__":
    main()
