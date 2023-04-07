import numpy as np
import torch

# from sklearn.metrics import average_precision_score


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


def get_iou_backup(x_hat_arr, y_hat_arr, w_hat_arr, h_hat_arr, bbox):
    x1_hat_arr = x_hat_arr - (w_hat_arr / 2)
    x2_hat_arr = x_hat_arr + (w_hat_arr / 2)
    y1_hat_arr = y_hat_arr - (h_hat_arr / 2)
    y2_hat_arr = y_hat_arr + (h_hat_arr / 2)

    intersection_arr = (
        np.maximum(
            np.minimum(bbox.x2, x2_hat_arr) -
            np.maximum(bbox.x1, x1_hat_arr),
            0
        ) *
        np.maximum(
            np.minimum(bbox.y2, y2_hat_arr) -
            np.maximum(bbox.y1, y1_hat_arr),
            0
        )
    )
    union_arr = (
        (x2_hat_arr - x1_hat_arr) * (y2_hat_arr - y1_hat_arr) +
        (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1) -
        intersection_arr
    )

    iou_arr = intersection_arr / union_arr

    return iou_arr


def get_max_iou_indices(x_hat_arr, y_hat_arr, w_hat_arr, h_hat_arr, bbox):
    iou_arr = get_iou(x_hat_arr, y_hat_arr, w_hat_arr, h_hat_arr, bbox)

    max_iou = np.max(iou_arr)

    return iou_arr, max_iou, np.argmax(iou_arr) * 5 + np.array([0, 1, 2, 3, 4])


def cummax(x):
    return np.array([np.max(x[:i + 1]) for i in range(len(x))])


def get_aps(
    iou_batch,
    cls_tgt_batch,
    cls_score_batch,
    img_id_batch,
    level_list=[.50, .55, .60, .65, .70, .75, .80, .85, .90, .95],
):
    '''
        Args:
            iou_batch: [N, S, S, B]
            cls_tgt_batch: [N, C]
            cls_score_batch: [N, S, S, B, C]
            img_id_batch: [N]
    '''
    # N = iou_batch.shape[0]
    # S = iou_batch.shape[1]
    # B = iou_batch.shape[-1]
    C = cls_tgt_batch.shape[-1]

    aps = {}

    for level in level_list:

        aps_by_class = []

        for cls_idx in range(C):
            # selected_cls_indices: [M]
            selected_cls_indices = np.where(cls_tgt_batch[:, cls_idx] == 1)[0]

            # selected_iou_batch: [M, S, S, B]
            # selected_cls_tgt_batch: [M]
            # selected_cls_score_batch: [M, S, S, B]
            # selected_img_id_batch: [M]
            selected_iou_batch = iou_batch[selected_cls_indices]
            # selected_cls_tgt_batch = cls_tgt_batch[
            #     selected_cls_indices, cls_idx
            # ]
            selected_cls_score_batch = cls_score_batch[
                selected_cls_indices, :, :, :, cls_idx
            ]
            selected_img_id_batch = img_id_batch[selected_cls_indices]

            num_gt = selected_iou_batch.shape[0]

            fp_list = []
            tp_list = []
            score_list = []

            for img_id in np.unique(selected_img_id_batch):
                # one_img_indices: [L]
                one_img_indices = np.where(selected_img_id_batch == img_id)

                # one_img_iou_batch: [L, S, S, B]
                # one_img_cls_score_batch: [L, S, S, B] -> [S, S, B]
                one_img_iou_batch = selected_iou_batch[
                    one_img_indices
                ]
                one_img_cls_score_batch = selected_cls_score_batch[
                    one_img_indices
                ]
                one_img_cls_score_batch = one_img_cls_score_batch[0]

                # one_img_max_iou_by_grid_cell_batch: [S, S, B]
                one_img_max_iou_by_grid_cell_batch = np.max(
                    one_img_iou_batch, axis=0
                )

                # fp, tp: [S, S, B]
                fp = (one_img_max_iou_by_grid_cell_batch < level)
                tp = (one_img_max_iou_by_grid_cell_batch >= level)

                fp_list.append(fp.flatten())
                tp_list.append(tp.flatten())
                score_list.append(one_img_cls_score_batch.flatten())

            # fp_list, tp_list, score_list: [M]
            fp_list = np.hstack(fp_list)
            tp_list = np.hstack(tp_list)
            score_list = np.hstack(score_list)

            sorted_indices = np.argsort(score_list)

            sorted_fp_list = fp_list[sorted_indices]
            sorted_tp_list = tp_list[sorted_indices]
            # sorted_score_list = score_list[sorted_indices]

            cum_fp_list = np.cumsum(sorted_fp_list)
            cum_tp_list = np.cumsum(sorted_tp_list)

            prec_list = cum_tp_list / (cum_tp_list + cum_fp_list)
            rec_list = cum_tp_list / num_gt

            reverse_cummax_prec_list = cummax(prec_list[::-1])[::-1]

            rec_diff_list = rec_list - np.hstack([[0], rec_list[:-1]])

            ap = np.sum(reverse_cummax_prec_list * rec_diff_list)

            aps_by_class.append(ap)

        aps["APs by Class"] = aps_by_class
        aps[level] = np.mean(aps_by_class)

    mean_ap = np.mean([aps[level] for level in level_list])
    aps["mAP"] = mean_ap

    return aps

    # for level in level_list:

    #     aps_by_class = []

    #     for cls_idx in range(C):
    #         # selected_cls_indices: [L']
    #         # This selection would eliminate the true nagatives also.
    #         selected_cls_indices = (
    #             np.where(cls_tgt_arr_list[:, cls_idx] == 1)[0]
    #         )

    #         # selected_iou_list: [L']
    #         # selected_cls_score_arr_list: [L']
    #         # selected_cls_tgt_arr_list: [L']
    #         selected_iou_list = iou_list[selected_cls_indices, 0]
    #         selected_cls_tgt_list = class_true_arr_list[
    #             selected_class_indices, cls_idx
    #         ]
    #         selected_class_score_list = class_score_arr_list[
    #             selected_class_indices, cls_idx
    #         ]

    #         selected_class_true_list[
    #             np.where(selected_iou_list < level)[0]
    #         ] = 0

    #         ap_by_class = average_precision_score(
    #             selected_class_true_list,
    #             selected_class_score_list,
    #         )

    #         aps_by_class.append(ap_by_class)

    #     aps[level] = np.mean(aps_by_class)

    # mean_ap = np.mean([v for v in aps.values()])
    # aps["mAP"] = mean_ap

    # return aps

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
