import numpy as np
import torch
import imgaug as ia
import imgaug.augmenters as iaa


def get_augmenter(height, width):
    augmenter = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width
            #   (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order:
            #   - use nearest neighbour or bilinear interpolation (fast)
            # - mode:
            #   - use any available mode to fill newly created pixels
            #   - see API or scikit-image for which modes are available
            # - cval:
            #   - if the mode is constant, then use a random brightness
            #   for the newly created pixels (e.g. sometimes black,
            #   sometimes white)
            iaa.Sometimes(0.5, iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters
            # per image. Don't execute all of them, as that would often be
            # way too strong.
            #
            iaa.SomeOf(
                (0, 5),
                [
                    # Convert some images into their superpixel
                    # representation, sample between 20 and 200
                    # superpixels per image, but do not replace all
                    # superpixels with their average, only some of them
                    # (p_replace).
                    iaa.Sometimes(
                        0.5,
                        iaa.Superpixels(
                            p_replace=(0, 1.0),
                            n_segments=(20, 200)
                        )
                    ),

                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur
                    # (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ]),

                    # Sharpen each image, overlay the result with the
                    # original image using an alpha between 0 (no
                    # sharpening) and 1
                    # (full sharpening effect).
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                    # Same as sharpen, but for an embossing effect.
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                    # Search in some images either for all edges or for
                    # directed edges. These edges are then marked in a
                    # black and white image and overlayed with the
                    # original image using an alpha of 0 to 0.7.
                    iaa.Sometimes(
                        0.5,
                        iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0, 0.7)),
                            iaa.DirectedEdgeDetect(
                                alpha=(0, 0.7), direction=(0.0, 1.0)
                            ),
                        ])
                    ),

                    # Add gaussian noise to some images.
                    # In 50% of these cases, the noise is randomly sampled
                    # per channel and pixel.
                    # In the other 50% of all cases it is sampled once per
                    # pixel (i.e. brightness change).
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                    ),

                    # Either drop randomly 1 to 10% of all pixels (i.e. set
                    # them to black) or drop them on an image with 2-5%
                    # percent of the original size, leading to large
                    # dropped rectangles.
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.03, 0.15), size_percent=(0.02, 0.05),
                            per_channel=0.2
                        ),
                    ]),

                    # Invert each image's channel with 5% probability.
                    # This sets each pixel value v to 255-v.
                    iaa.Invert(0.05, per_channel=True),

                    # Add a value of -10 to 10 to each pixel.
                    iaa.Add((-10, 10), per_channel=0.5),

                    # Change brightness of images (50-150% of original
                    # value).
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),

                    # Improve or worsen the contrast of images.
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                    # Convert each image to grayscale and then overlay the
                    # result with the original with random alpha. I.e.
                    # remove colors with varying strengths.
                    iaa.Grayscale(alpha=(0.0, 1.0)),

                    # In some images move pixels locally around (with
                    # random strengths).
                    iaa.Sometimes(
                        0.5,
                        iaa.ElasticTransformation(
                            alpha=(0.5, 3.5), sigma=0.25
                        )
                    ),

                    # In some images distort local areas with varying
                    # strength.
                    iaa.Sometimes(
                        0.5,
                        iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                ],
                # do all of the above augmentations in random order
                random_order=True
            )
        ],
        random_order=True
    )

    resize = iaa.Resize({"height": height, "width": width})

    return augmenter, resize


def get_iou(
    x1_batch,
    x2_batch,
    y1_batch,
    y2_batch,
    x1_hat_batch,
    x2_hat_batch,
    y1_hat_batch,
    y2_hat_batch,
):
    '''
        Returns the IOU batch from x1, x2, y1, y2, x1_hat, x2_hat, y1_hat,
        y2_hat batches

        Args:
            x1_batch:
                - [N, ...]
            x2_batch:
                - [N, ...]
            y1_batch:
                - [N, ...]
            y2_batch:
                - [N, ...]
            x1_hat_batch:
                - [N, ...]
            x2_hat_batch:
                - [N, ...]
            y1_hat_batch:
                - [N, ...]
            y2_hat_batch:
                - [N, ...]

        Returns:
            iou_batch:
                - [N, ...]
    '''
    intsec_x1_batch = torch.maximum(x1_batch, x1_hat_batch)
    intsec_y1_batch = torch.maximum(y1_batch, y1_hat_batch)
    intsec_x2_batch = torch.minimum(x2_batch, x2_hat_batch)
    intsec_y2_batch = torch.minimum(y2_batch, y2_hat_batch)

    intsec_batch = (
        torch.clamp(intsec_x2_batch - intsec_x1_batch, min=0) *
        torch.clamp(intsec_y2_batch - intsec_y1_batch, min=0)
    )

    union_batch = (
        (x2_batch - x1_batch) * (y2_batch - y1_batch) +
        (x2_hat_batch - x1_hat_batch) * (y2_hat_batch - y1_hat_batch) -
        intsec_batch
    )

    iou_batch = intsec_batch / (union_batch + 1e-6)

    return iou_batch


def nms(
    x1_batch,
    x2_batch,
    y1_batch,
    y2_batch,
    conf_score_batch,
    conf_score_thre=0.6,
    iou_thre=0.5
):
    '''
        Args:
            x1_batch:
                - torch.Tensor
                - [N, ...]
            x2_batch:
                - torch.Tensor
                - [N, ...]
            y1_batch:
                - torch.Tensor
                - [N, ...]
            y2_batch:
                - torch.Tensor
                - [N, ...]
            conf_score_batch:
                - torch.Tensor
                - [N, ...]
    '''

    # conf_score_mask_batch: [N, ...]
    conf_score_mask_batch = (conf_score_batch >= conf_score_thre).bool()

    # x1_batch, x2_batch, y1_batch, y2_batch, conf_score_batch: [M]
    x1_batch = torch.masked_select(x1_batch, conf_score_mask_batch)
    x2_batch = torch.masked_select(x2_batch, conf_score_mask_batch)
    y1_batch = torch.masked_select(y1_batch, conf_score_mask_batch)
    y2_batch = torch.masked_select(y2_batch, conf_score_mask_batch)
    conf_score_batch = torch.masked_select(
        conf_score_batch, conf_score_mask_batch
    )

    # for sorting the coordinates.
    conf_score_batch, sorted_indices = torch.sort(
        conf_score_batch, descending=True
    )

    x1_batch = x1_batch[sorted_indices]
    x2_batch = x2_batch[sorted_indices]
    y1_batch = y1_batch[sorted_indices]
    y2_batch = y2_batch[sorted_indices]

    i = 0
    while i < len(conf_score_batch) - 1:
        # x1_tgt, x2_tgt, y1_tgt, y2_tgt: [1]
        x1_tgt = x1_batch[i:i + 1]
        x2_tgt = x2_batch[i:i + 1]
        y1_tgt = y1_batch[i:i + 1]
        y2_tgt = y2_batch[i:i + 1]

        # iou_batch: [M - i]
        iou_batch = get_iou(
            x1_tgt,
            x2_tgt,
            y1_tgt,
            y2_tgt,
            x1_batch[i + 1:],
            x2_batch[i + 1:],
            y1_batch[i + 1:],
            y2_batch[i + 1:],
        )

        # iou_mask: [M - i]
        iou_mask = (iou_batch >= iou_thre).bool()

        # iou_mask: [M]
        iou_mask = torch.cat([torch.tensor([True] * i), iou_mask], dim=-1)

        # x1_batch, x2_batch, y1_batch, y2_batch, conf_score_batch: [M']
        x1_batch = torch.masked_select(x1_batch, iou_mask)
        x2_batch = torch.masked_select(x2_batch, iou_mask)
        y1_batch = torch.masked_select(y1_batch, iou_mask)
        y2_batch = torch.masked_select(y2_batch, iou_mask)
        conf_score_batch = torch.masked_select(conf_score_batch, iou_mask)

    return x1_batch, x2_batch, y1_batch, y2_batch, conf_score_batch


def cummax(x):
    return np.array([np.max(x[:i + 1]) for i in range(len(x))])


def get_aps(
    iou_batch,
    cls_tgt_batch,
    cls_score_batch,
    bbox_img_id_batch,
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
            selected_img_id_batch = bbox_img_id_batch[selected_cls_indices]

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

            sorted_indices = np.argsort(score_list)[::-1]

            sorted_fp_list = fp_list[sorted_indices]
            sorted_tp_list = tp_list[sorted_indices]
            # sorted_score_list = score_list[sorted_indices]

            cum_fp_list = np.cumsum(sorted_fp_list)
            cum_tp_list = np.cumsum(sorted_tp_list)

            prec_list = cum_tp_list / (cum_tp_list + cum_fp_list)
            rec_list = cum_tp_list / num_gt

            reverse_cummax_prec_list = cummax(prec_list[::-1])[::-1]

            rec_diff_list = rec_list - np.hstack([[0], rec_list[:-1]])

            # print("==============================================")
            # print(cls_idx)
            # print("sorted_score_list", sorted_score_list)
            # print("prec_list", prec_list)
            # print("rec_list", rec_list)
            # print("reverse_cummax_prec_list", reverse_cummax_prec_list)
            # print("rec_diff_list", rec_diff_list)

            ap = np.sum(reverse_cummax_prec_list * rec_diff_list)

            # print("ap", ap)

            aps_by_class.append(ap)

        aps["APs by Class"] = aps_by_class
        aps[level] = np.mean(aps_by_class)

    mean_ap = np.mean([aps[level] for level in level_list])
    aps["mAP"] = mean_ap

    return aps
