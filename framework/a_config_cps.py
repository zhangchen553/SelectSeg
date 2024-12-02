import glob
from sklearn.model_selection import train_test_split
from imutils import paths
import os
import sys

# ----------------------    Configuration-------------------------------------------
platform = 'windows'  # 'linux', 'windows'
current_dir = os.path.dirname(__file__)
STORAGE_PATH = os.path.dirname(current_dir)
sys.path.append(STORAGE_PATH)

VAL_IMG_PATH = False
VAL_MASK_PATH = False
TEST_IMG_PATH = False
TEST_MASK_PATH = False

# Training configuration-----------------------------------------
# ---------------------------------------------------------------
DEVICE = 'cuda:0'  # if torch.cuda.is_available() else 'cpu'
WORKER = 4

# This lets your DataLoader allocate the samples in
# page-locked memory, which speeds-up the data transfer.
PIN_MEMORY = True  # if DEVICE == 'cuda:0' else False

# Fraction of test dataset
VAL_SPLIT = 0.001
TEST_SPLIT = 0.2

INPUT_IMAGE_WIDTH = 320
INPUT_IMAGE_HEIGHT = 320

NUM_CLASSES = 1

DATASET = 'CRACK500'  # 'DEEPCRACK'#'CRACK500' 'TOPO' 'HKCrack'
BACKBONE = 'mit_b2'  # 'resnet50'， 'mit_b2'
DECODER = 'FPN'  # 'DeepLabV3Plus' #“Unet”
AUG = 'base'  # trivial or base, 'randaug' 'gaussianvvar' augmix, clahe
LOSSFN = 'Dice'  # 'Dice' Focal BCE

if DATASET == 'CRACK500':
    BATCH_SIZE = 16
    INIT_LR = 2e-4
else:
    BATCH_SIZE = 8
    INIT_LR = 1e-4

save_folder = '/Results_' + DATASET + '/cps_new/'

# if ('mit' in BACKBONE) or ('convnext' in BACKBONE):
#     INIT_LR = 5e-5  # 1.5e-4, 5e-5
# else:
#     INIT_LR = 1.5e-4  # 3e-4 1.5e-4

SEED = 3

MEAN = (0, 0, 0)
STD = (1, 1, 1)

SLACK = 3
MAX_epochs = 300  # 30000

# ------------partial fully and Semi-supervised learning setting----------
semi = True
partial_fully = False
train_test_portion = [100, 99, 98, 95, 90, 80, 50]  # pct
sup_portion = 60
unsup_portion = 100 - sup_portion
partial_type = ''  # '_f1_base' '_mse_base' ''_f1_f1_var

c_ratio = 50
if DATASET == 'CRACK500':
    DATA_PATH = STORAGE_PATH + '/DATASET/CRACK500'
    TRAIN_IMG_PATH = DATA_PATH + '/train_crop_image/'
    TRAIN_MASK_PATH = DATA_PATH + '/train_crop_mask_' + str(c_ratio) + 'pct/'
    VAL_IMG_PATH = DATA_PATH + '/val_crop_image/'
    VAL_MASK_PATH = DATA_PATH + '/val_crop_mask/'
    TEST_IMG_PATH = DATA_PATH + '/test_crop_image/'
    TEST_MASK_PATH = DATA_PATH + '/test_crop_mask/'
    INPUT_IMAGE_WIDTH = 320
    INPUT_IMAGE_HEIGHT = 320
    FLAG = 'val'

    MEAN = (0.4693, 0.4713, 0.4723)
    STD = (0.1898, 0.1883, 0.1855)

# Treat validation image as test image since we save model according to test, and we only have val data
if DATASET == 'HKCrack':
    DATA_PATH = STORAGE_PATH + '/DATASET/HKCrack'
    TRAIN_IMG_PATH = DATA_PATH + '/train_crop_image_v2/'  # /better_train_img_20/, /train_crop_image_v2/
    TRAIN_MASK_PATH = DATA_PATH + '/train_crop_mask_v2/'  # /better_train_mask_20/, /train_crop_mask_v2/
    VAL_IMG_PATH = DATA_PATH + '/val_crop_image/'
    VAL_MASK_PATH = DATA_PATH + '/val_crop_mask/'
    TEST_IMG_PATH = DATA_PATH + '/test_crop_image/'
    TEST_MASK_PATH = DATA_PATH + '/test_crop_mask_v3/'  # val_crop_mask_v2
    INPUT_IMAGE_WIDTH = 512
    INPUT_IMAGE_HEIGHT = 512
    FLAG = 'val'

    MEAN = (0.4937, 0.4957, 0.4967)
    STD = (0.1656, 0.1635, 0.1599)

# Image mask in same folder but different format
trainImages = glob.glob(r'{}*.jpg'.format(TRAIN_IMG_PATH))  # or jpg, jpeg
trainMasks = glob.glob(r'{}*.png'.format(TRAIN_MASK_PATH))

# Image mask in different folder, but train image in png format
if not trainImages:
    trainImages = sorted(list(paths.list_images(TRAIN_IMG_PATH)))
    trainMasks = sorted(list(paths.list_images(TRAIN_MASK_PATH)))

# dataset has test images
if TEST_IMG_PATH:
    testImages = glob.glob(r'{}*.jpg'.format(TEST_IMG_PATH))  # or jpg, jpeg
    testMasks = glob.glob(r'{}*.png'.format(TEST_MASK_PATH))
    if testImages == []:
        testImages = sorted(list(paths.list_images(TEST_IMG_PATH)))
        testMasks = sorted(list(paths.list_images(TEST_MASK_PATH)))
else:
    split = train_test_split(trainImages, trainMasks, test_size=TEST_SPLIT, random_state=42)
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

# dataset has validation images
if VAL_IMG_PATH:
    valImages = glob.glob(r'{}*.jpg'.format(VAL_IMG_PATH)) # or jpg, jpeg
    valMasks = glob.glob(r'{}*.png'.format(VAL_MASK_PATH))
    if valImages ==[]:
        valImages = sorted(list(paths.list_images(VAL_IMG_PATH)))
        valMasks = sorted(list(paths.list_images(VAL_MASK_PATH)))
else:
    split = train_test_split(trainImages, trainMasks,
                             test_size=VAL_SPLIT, random_state=42)
    
    (trainImages, valImages) = split[:2]
    (trainMasks, valMasks) = split[2:]


#  ----------------------------selective training validation configuration--------------------------------

data_output_path = STORAGE_PATH + '/Results_' + DATASET + '/fully_supervise/data_split_txt/'



