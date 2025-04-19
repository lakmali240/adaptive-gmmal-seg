This is for self-supervised learning
------------------------------------

1) Change the following parameters to match with your self-supervised training in self_supervised_learning.py

""" Training Hyperparameters """
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
START_EPOCH = 1
NUM_EPOCHS = 100
NUM_WORKERS = 4
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
PIN_MEMORY = True

""" Early Stopping Conditions """
EARLY_STOPPING_EPOCHES = 30
EXPECTED_BEST_LOSS = 0.005

2) If you already have a pre-trained model, 

""" Load Pre-trained Model"""
LOAD_TRAINED_MODEL = True
PATH_TO_TRAINED_MODEL = '<pre-trained model path>'

Otherwise, 
""" Load Pre-trained Model"""
LOAD_TRAINED_MODEL = Flase
PATH_TO_TRAINED_MODEL = ' '

3) Traning and validation dataset directory

TRAIN_IMG_DIR = "<training image directory>"
TRAIN_MASK_DIR = "<training mask directory>" # not used here
VAL_IMG_DIR = "<validation image directory>"
VAL_MASK_DIR = "<validation mask directory>" # not used here


