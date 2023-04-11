# You can choose device config from: cpu, cuda, mps
DEVICE = "cpu"

BATCH_SIZE = 64

MODEL_CONFIG = {
    "YOLOv1": {
        "S": 7,
        "B": 2,
    },
    "YOLOv2": {
        "S": 13,
    },
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
}
