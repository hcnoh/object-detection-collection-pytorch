import numpy as np
import torch

from sklearn.metrics import average_precision_score


def get_iou(
    x1_arr,
    x2_arr,
    y1_arr,
    y2_arr,
    x1_hat_arr,
    x2_hat_arr,
    y1_hat_arr,
    y2_hat_arr,
):
    intersection_arr = (
        torch.relu(
            torch.minimum(x2_arr, x2_hat_arr) -
            torch.maximum(x1_arr, x1_hat_arr),
        ) *
        torch.relu(
            torch.minimum(y2_arr, y2_hat_arr) -
            torch.maximum(y1_arr, y1_hat_arr),
        )
    )
    union_arr = (
        (x2_hat_arr - x1_hat_arr) * (y2_hat_arr - y1_hat_arr) +
        (x2_arr - x1_arr) * (y2_arr - y1_arr) -
        intersection_arr
    )

    iou_arr = intersection_arr / union_arr

    return iou_arr


def get_iou_backup(x_hat_arr, y_hat_arr, w_hat_arr, h_hat_arr, bndbox):
    x1_hat_arr = x_hat_arr - (w_hat_arr / 2)
    x2_hat_arr = x_hat_arr + (w_hat_arr / 2)
    y1_hat_arr = y_hat_arr - (h_hat_arr / 2)
    y2_hat_arr = y_hat_arr + (h_hat_arr / 2)

    intersection_arr = (
        np.maximum(
            np.minimum(bndbox.x2, x2_hat_arr) -
            np.maximum(bndbox.x1, x1_hat_arr),
            0
        ) *
        np.maximum(
            np.minimum(bndbox.y2, y2_hat_arr) -
            np.maximum(bndbox.y1, y1_hat_arr),
            0
        )
    )
    union_arr = (
        (x2_hat_arr - x1_hat_arr) * (y2_hat_arr - y1_hat_arr) +
        (bndbox.x2 - bndbox.x1) * (bndbox.y2 - bndbox.y1) -
        intersection_arr
    )

    iou_arr = intersection_arr / union_arr

    return iou_arr


def get_max_iou_indices(x_hat_arr, y_hat_arr, w_hat_arr, h_hat_arr, bndbox):
    iou_arr = get_iou(x_hat_arr, y_hat_arr, w_hat_arr, h_hat_arr, bndbox)

    max_iou = np.max(iou_arr)

    return iou_arr, max_iou, np.argmax(iou_arr) * 5 + np.array([0, 1, 2, 3, 4])


def get_aps(
    iou_list,
    class_true_arr_list,
    class_score_arr_list,
    level_list=[.50, .55, .60, .65, .70, .75, .80, .85, .90, .95],
):
    '''
        Args:
            iou_list: [L, 1]
            class_true_arr_list: [L, C]
            class_score_arr_list: [L, C]
    '''
    C = class_true_arr_list.shape[-1]

    aps = {}

    for level in level_list:

        aps_by_class = []

        for cls_idx in range(C):
            # selected_class_indices: [L']
            selected_class_indices = (
                np.where(class_true_arr_list[:, cls_idx] == 1)[0]
            )

            # selected_iou_list: [L']
            # selected_class_true_arr: [L']
            # selected_class_score_list: [L']
            selected_iou_list = iou_list[selected_class_indices]
            selected_class_true_list = class_true_arr_list[
                selected_class_indices
            ]
            selected_class_score_list = class_score_arr_list[
                selected_class_indices
            ]

            selected_class_score_list[
                np.where(selected_iou_list < level)[0]
            ] = 0

            ap_by_class = average_precision_score(
                selected_class_true_list,
                selected_class_score_list,
            )

            aps_by_class.append(ap_by_class)

        aps[level] = np.mean(aps_by_class)

    mean_ap = np.mean([v for v in aps.values()])
    aps["mAP"] = mean_ap

    return aps

    # for level in level_list:
    #     mask = (iou_list >= level)

    #     if np.sum(mask) != 0:
    #         masked_class_true_arr_list = (
    #             class_true_arr_list[np.where(mask == 1)[0]]
    #         )
    #         masked_class_score_arr_list = (
    #             class_score_arr_list[np.where(mask == 1)[0]]
    #         )

    #         AP = average_precision_score(
    #             masked_class_true_arr_list,
    #             masked_class_score_arr_list,
    #             average="macro",
    #         )

    #         aps[level] = AP

    #     else:
    #         aps[level] = 0

    # mAP = np.mean([v for v in aps.values()])
    # aps["mAP"] = mAP

    # return aps
