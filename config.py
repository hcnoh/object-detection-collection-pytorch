# You can choose device config from: cpu, cuda, mps
DEVICE = "cpu"

BATCH_SIZE = 16

MODEL_CONFIG = {
    "YOLOv1": {
        "S": 7,
        "B": 2,
    },
    "YOLOv2": {},
}

TRAIN_CONFIG = {
    "YOLOv1": {
        "VOC2012": {
            "learning_rate_list": [
                1e-2,
                1e-3,
                1e-4,
            ],
            "num_epochs_list": [
                500,
                300,
                300,
            ],
            "lambda_coord": 5,
            "lambda_noobj": 5e-1,
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
            "lambda_xy": 0.5,
            "lambda_wh": 0.5,
            "lambda_conf": 1.0,
            "lambda_noobj": 1.0,
            "lambda_cls": 1.0,
        }
    },
}
