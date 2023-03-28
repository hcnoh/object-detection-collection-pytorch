# You can choose device config from: cpu, cuda, mps
DEVICE = "cpu"

BATCH_SIZE = 128

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
                3e-3,
                1e-3,
                3e-4,
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
