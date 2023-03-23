# You can choose device config from: cpu, cuda, mps
DEVICE = "cpu"

MODEL_CONFIG = {
    "YOLOv1": {
        "S": 7,
        "B": 2,
    }
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
                75,
                30,
                30,
            ],
            "lambda_coord": 5,
            "lambda_noobj": .5,
        }
    }
}
