from imports import *


warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


SEED = 2019

DATA_PATH = '../input/VOC2007/'

IMG_PATH = DATA_PATH + 'JPEGImages/'
ANNOTATION_PATH = DATA_PATH + 'Annotations/'

CP_PATH = f'../checkpoints/{date.today()}/'

TRAIN_IMGS = DATA_PATH + "ImageSets/Layout/train.txt"
TRAINVAL_IMGS = DATA_PATH + "ImageSets/Layout/trainval.txt"
VAL_IMGS = DATA_PATH + "ImageSets/Layout/val.txt"
TEST_IMGS = DATA_PATH + "ImageSets/Layout/test.txt"

GLOVE_PATH = '../input/glove/'
GLOVE_DIM = 300

VISUAL_GENOME_PATH  = '../input/visual_genome/

CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor'
        ]
NUM_CLASSES = len(CLASSES)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

NUM_WORKERS = 4
VAL_BS = 32

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

IMG_SIZE = 224
# IMG_SIZE = 576