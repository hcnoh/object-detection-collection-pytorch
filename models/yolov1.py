import os
import pickle

import numpy as np
import torch
import torch.cuda
import torch.backends.mps
# import imgaug as ia
import imgaug.augmenters as iaa

from torch.nn import Module, Sequential, Flatten, Linear, ReLU
from torch.optim import Adam, SGD

from config import DEVICE
from models.backbones.googlenet import GoogLeNetBackbone
from models.utils import get_max_iou_indices, get_aps


class YOLOv1(Module):
    def __init__(
        self,
        S,
        B,
        clc_list,
        clc2idx,
    ) -> None:
        super().__init__()

        self.backbone_model = GoogLeNetBackbone()

        self.h_in = self.backbone_model.h_in
        self.w_in = self.backbone_model.w_in

        self.S = S
        self.B = B

        self.clc_list = clc_list
        self.clc2idx = clc2idx

        self.C = len(self.clc_list)

        self.backbone_output_dim = np.prod(self.backbone_model.output_shape)

        self.linear_layers = Sequential(
            Flatten(),
            Linear(self.backbone_output_dim, 4096),
            ReLU(),
            Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )

    def forward(self, x):
        h = self.backbone_model(x)
        h = self.linear_layers(h)
        h = h.reshape([-1, self.S, self.S, self.B * 5 + self.C])
        y = torch.sigmoid(h)

        return y

    def get_loss(
        self,
        N,
        y_pred_batch,
        bndbox_arr_list,
        lambda_coord,
        lambda_noobj,
    ):
        # pc_arr: [N, S, S, C]
        pc_arr = np.zeros(shape=[N, self.S, self.S, self.C])

        # noobj_mask_batch: [N, S, S, B]
        noobj_mask_batch = np.ones(shape=[N, self.S, self.S, self.B])

        loss = []
        iou_list = []
        class_true_arr_list = []
        class_score_arr_list = []

        for n in range(N):
            for bndbox in bndbox_arr_list[n]:
                bx = (bndbox.x1 + bndbox.x2) / 2
                by = (bndbox.y1 + bndbox.y2) / 2

                i = int(by // (self.h_in / self.S))
                j = int(bx // (self.w_in / self.S))

                noobj_mask_batch[n, i, j, :] = 0

                # bx_pred_arr, by_pred_arr, bw_pred_arr, bh_pred_arr: [B * 5]
                bx_pred_arr = (
                    y_pred_batch[n, i, j, 0:self.B * 5:5] *
                    (self.w_in / self.S) +
                    i * (self.w_in / self.S)
                ).detach().cpu().numpy()

                by_pred_arr = (
                    y_pred_batch[n, i, j, 1:self.B * 5:5] *
                    (self.h_in / self.S) +
                    j * (self.h_in / self.S)
                ).detach().cpu().numpy()

                bw_pred_arr = (
                    y_pred_batch[n, i, j, 2:self.B * 5:5] * self.w_in
                ).detach().cpu().numpy()

                bh_pred_arr = (
                    y_pred_batch[n, i, j, 3:self.B * 5:5] * self.h_in
                ).detach().cpu().numpy()

                # iou_arr: [B]
                # max_iou_indices: [B * 5]
                iou_arr, max_iou, max_iou_indices = get_max_iou_indices(
                    bx_pred_arr,
                    by_pred_arr,
                    bw_pred_arr,
                    bh_pred_arr,
                    bndbox,
                )

                bx = (
                    torch.tensor(
                        (bx % (self.w_in / self.S)) /
                        (self.w_in / self.S)
                    )
                    .float().to(DEVICE)
                )
                by = (
                    torch.tensor(
                        (by % (self.h_in / self.S)) /
                        (self.h_in / self.S)
                    )
                    .float().to(DEVICE)
                )
                bw = (
                    torch.tensor(
                        (bndbox.x2 - bndbox.x1) / self.w_in
                    )
                    .float().to(DEVICE)
                )
                bh = (
                    torch.tensor(
                        (bndbox.y2 - bndbox.y1) / self.h_in
                    )
                    .float().to(DEVICE)
                )
                c = torch.tensor(max_iou).to(DEVICE)
                pc_arr[n, i, j, self.clc2idx[bndbox.label]] = 1

                bx_pred = y_pred_batch[n, i, j, max_iou_indices[0]]
                by_pred = y_pred_batch[n, i, j, max_iou_indices[1]]
                bw_pred = y_pred_batch[n, i, j, max_iou_indices[2]]
                bh_pred = y_pred_batch[n, i, j, max_iou_indices[3]]
                c_pred = y_pred_batch[n, i, j, max_iou_indices[4]]

                loss.append(
                    lambda_coord * (
                        (bx - bx_pred) ** 2 + (by - by_pred) ** 2
                    ) +
                    lambda_coord * (
                        (torch.sqrt(bw) - torch.sqrt(bw_pred)) ** 2 +
                        (torch.sqrt(bh) - torch.sqrt(bh_pred)) ** 2
                    ) +
                    (c - c_pred) ** 2
                )

                class_true_arr = np.repeat(
                    np.expand_dims(
                        np.eye(self.C)[self.clc2idx[bndbox.label]], axis=0
                    ),
                    repeats=self.B,
                    axis=0,
                )

                class_score_arr = np.repeat(
                    np.expand_dims(
                        y_pred_batch[n, i, j, -self.C:].detach().cpu().numpy(),
                        axis=0
                    ),
                    repeats=self.B,
                    axis=0,
                )

                iou_list.append(iou_arr)
                class_true_arr_list.append(class_true_arr)
                class_score_arr_list.append(class_score_arr)

        if len(loss) != 0:
            loss = torch.stack(loss, dim=0).mean()
        else:
            loss = torch.tensor(0.).to(DEVICE)

        # pc_arr, pc_pred_arr: [N, S, S, C]
        pc_arr = torch.tensor(pc_arr).float().to(DEVICE)
        pc_pred_arr = y_pred_batch[:, :, :, -self.C:]

        loss += ((pc_arr - pc_pred_arr) ** 2).mean()

        loss += lambda_noobj * (
            torch.masked_select(
                y_pred_batch[:, :, :, 4:self.B * 5:5],
                mask=torch.tensor(noobj_mask_batch).bool().to(DEVICE)
            )
        ).mean()

        # iou_list: [L, B]
        # class_true_arr_list, class_score_arr_list: [L, B, C]
        iou_list = np.vstack(iou_list)
        class_true_arr_list = np.vstack(
            np.expand_dims(class_true_arr_list, axis=0)
        )
        class_score_arr_list = np.vstack(
            np.expand_dims(class_score_arr_list, axis=0)
        )

        return loss, iou_list, class_true_arr_list, class_score_arr_list

    def train_one_step(
        self,
        epoch,
        train_loader,
        lambda_coord,
        lambda_noobj,
        opt,
    ):
        train_loss_mean = []
        train_iou_list = []
        train_class_true_arr_list = []
        train_class_score_arr_list = []

        train_size = len(train_loader.dataset)
        progress_size = 0

        for data in train_loader:
            # x_batch: [N, H, W, 3]
            x_batch, lbl_batch = data

            N = x_batch.shape[0]
            progress_size += N

            print(
                "Epoch: {} --> [{} / {}]"
                .format(epoch, progress_size, train_size),
                end="\r"
            )

            self.train()

            # y_pred_batch: [N, S, S, B * 5 + C]
            y_pred_batch = self(x_batch)

            (
                loss,
                iou_list,
                class_true_arr_list,
                class_score_arr_list
            ) = self.get_loss(
                N, y_pred_batch, lbl_batch, lambda_coord, lambda_noobj
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_mean.append(loss.detach().cpu().numpy())
            train_iou_list.append(iou_list)
            train_class_true_arr_list.append(class_true_arr_list)
            train_class_score_arr_list.append(class_score_arr_list)

        train_loss_mean = np.mean(train_loss_mean)

        # iou_list: [L, B]
        # class_true_arr_list, class_score_arr_list: [L, B, C]
        train_iou_list = np.vstack(train_iou_list)
        train_class_true_arr_list = np.vstack(train_class_true_arr_list)
        train_class_score_arr_list = np.vstack(train_class_score_arr_list)

        train_aps = get_aps(
            train_iou_list,
            train_class_true_arr_list,
            train_class_score_arr_list,
        )

        return train_loss_mean, train_aps

    def validate_one_step(
        self,
        epoch,
        val_loader,
        lambda_coord,
        lambda_noobj,
    ):
        val_loss_mean = []
        val_iou_list = []
        val_class_true_arr_list = []
        val_class_score_arr_list = []

        for data in val_loader:
            # x_batch: [N, H, W, 3]
            x_batch, lbl_batch = data

            N = x_batch.shape[0]

            self.eval()

            # y_pred_batch: [N, S, S, B * 5 + C]
            y_pred_batch = self(x_batch)

            (
                loss,
                iou_list,
                class_true_arr_list,
                class_score_arr_list
            ) = self.get_loss(
                N, y_pred_batch, lbl_batch, lambda_coord, lambda_noobj
            )

            val_loss_mean.append(loss.detach().cpu().numpy())
            val_iou_list.append(iou_list)
            val_class_true_arr_list.append(class_true_arr_list)
            val_class_score_arr_list.append(class_score_arr_list)

        val_loss_mean = np.mean(val_loss_mean)

        # iou_list: [L, B]
        # val_true_arr_list, val_score_arr_list: [L, B, C]
        val_iou_list = np.vstack(val_iou_list)
        val_class_true_arr_list = np.vstack(val_class_true_arr_list)
        val_class_score_arr_list = np.vstack(val_class_score_arr_list)

        val_aps = get_aps(
            val_iou_list,
            val_class_true_arr_list,
            val_class_score_arr_list,
        )

        return val_loss_mean, val_aps

    def train_model(
        self,
        train_loader,
        val_loader,
        learning_rate_list,
        num_epochs_list,
        lambda_coord,
        lambda_noobj,
        ckpt_path,
    ):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                val_loader: the PyTorch DataLoader instance for test
                num_epochs: the number of epochs
                opt: the optimization to train this model
                ckpt_path: the path to save this model's parameters
        '''
        cum_epoch = 0

        train_loss_mean_list = []
        train_aps_list = []

        val_loss_list = []
        val_aps_list = []

        min_val_loss = 1e+10

        for lr, num_epochs in zip(learning_rate_list, num_epochs_list):
            opt = Adam(self.parameters(), lr=lr)
            opt = SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=5e-4,
            )

            for epoch in range(1 + cum_epoch, num_epochs + 1 + cum_epoch):
                train_loss_mean, train_aps = self.train_one_step(
                    epoch, train_loader, lambda_coord, lambda_noobj, opt,
                )
                val_loss, val_aps = self.validate_one_step(
                    epoch, val_loader, lambda_coord, lambda_noobj,
                )

                print(
                    (
                        "Epoch: {} --> " +
                        "Training: (" +
                        "Loss Mean: {},  " +
                        "AP50: {},  " +
                        "AP75: {},  " +
                        "mAP: {}" +
                        ")    " +
                        "Validation: (" +
                        "Loss: {},  " +
                        "AP50: {},  " +
                        "AP75: {},  " +
                        "mAP: {}" +
                        ")"
                    )
                    .format(
                        epoch,
                        train_loss_mean,
                        train_aps[.50],
                        train_aps[.75],
                        train_aps["mAP"],
                        val_loss,
                        val_aps[.50],
                        val_aps[.75],
                        val_aps["mAP"],
                    )
                )

                train_loss_mean_list.append(train_loss_mean)
                train_aps_list.append(train_aps)

                val_loss_list.append(val_loss)
                val_aps_list.append(val_aps)

                if val_loss < min_val_loss:
                    torch.save(
                        self.state_dict(),
                        os.path.join(
                            ckpt_path, "best_model.ckpt"
                        )
                    )
                    min_val_loss = val_loss

            cum_epoch += num_epochs

        torch.save(
            self.state_dict(),
            os.path.join(
                ckpt_path, "final_model.ckpt"
            )
        )

        with open(
            os.path.join(ckpt_path, "training_result.pkl"),
            "wb",
        ) as f:
            pickle.dump(
                {
                    "train_loss_mean_list": train_loss_mean_list,
                    "train_aps_list": train_aps_list,
                    "val_loss_list": val_loss_list,
                    "val_aps_list": val_aps_list,
                },
                f
            )

    def collate_fn(self, batch):
        augmenter = iaa.Sequential([
            iaa.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            ),
            iaa.MultiplyHueAndSaturation(mul_saturation=(0.5, 1.5))
        ])
        resize = iaa.Resize({"height": self.h_in, "width": self.w_in})

        img_batch = []
        lbl_batch = []

        for img, lbl in batch:
            img = np.array(img)

            img_aug, lbl_aug = augmenter(image=img, bounding_boxes=lbl)

            lbl_aug = lbl_aug.remove_out_of_image().clip_out_of_image()

            img_aug, lbl_aug = resize(image=img_aug, bounding_boxes=lbl_aug)

            img_batch.append(torch.tensor(img_aug).to(DEVICE))
            lbl_batch.append(lbl_aug)

        img_batch = torch.stack(img_batch, dim=0)

        return img_batch, lbl_batch
