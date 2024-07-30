# constant configuration for application

DATA_HEADER = (
    '绝对时间',
    '流程相对时间',
    '步次相对时间',
    '步次序号',
    '步次类型',
    '电压',
    '电流',
    '线电压',
    '容量',
    '能量',
    '功率',
    '通道温度',
    '电池温度',
    '环境温度',
    '目标温度',
    '接触阻抗',
    'PWM'
)

AVAILABLE_DATA_HEADER = (
    '通道温度','电池温度', '目标温度','PWM'
)

TARGET_HEADER = 'PWM'

INPUT_SIZE = len(AVAILABLE_DATA_HEADER)
HIDDEN_SIZE = 50
OUTPUT_SIZE = 1
TEST_SIZE = 100

EPOCHS = 300
SAVE_EPOCH_COUNT = 100
SAVE_EPOCH_MODULE = 1

LEARNING_RATE = 0.001

TIME_STEP = 7
SPLIT_TRAIN_RATE = 0.7
SPLIT_VALID_RATE = 0.2
SPLIT_TEST_RATE = 0.1
SPLIT_RANDOM_STATE = 13891


LCM_CHANNELS_COUNT = 16
LCM_AVAILABLE_CHANNELS_COUNT = 4
LCM_CONTROL_INTERVAL = 1000


from pathlib import Path

MODEL_SAVED_FOLDER = Path('./saved_model/')
MODEL_EXPORT_FOLDER = Path('./export/')
del Path
