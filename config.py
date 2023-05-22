# You can choose device config from: cpu, cuda, mps
DEVICE = "cpu"

BATCH_SIZE = 16

MODEL_CONFIG = {
    "YOLOv1": {
        "num_grid_cell_in_height": 7,
        "num_grid_cell_in_width": 7,
        "num_anchor_box": 2,
    },
    "YOLOv2": {},
}

TRAIN_CONFIG = {
    "YOLOv1": {
        "VOC2012": {
            "learning_rate_list": [
                1e-3,
                1e-4,
                1e-5,
            ],
            "num_epoch_list": [
                60,
                30,
                70,
            ],
            "lambda_xy": 5.0,
            "lambda_wh": 5.0,
            "lambda_conf": 1.0,
            "lambda_noobj": 0.5,
            "lambda_cls": 1.0,
        }
    },
    "YOLOv2": {
        "VOC2012": {
            "learning_rate_list": [
                1e-3,
                1e-4,
                1e-5,
            ],
            "num_epoch_list": [
                60,
                30,
                70,
            ],
            "lambda_xy": 5.0,
            "lambda_wh": 5.0,
            "lambda_conf": 1.0,
            "lambda_noobj": 0.5,
            "lambda_cls": 1.0,
        }
    },
}
