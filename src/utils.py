# python native
import random

# external library
from PIL import Image
import numpy as np
from skimage.transform import resize

# torch
import torch

# seed setting
RANDOM_SEED = 666

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    
# for visualization -> 클래스가 2개 이상인 픽셀을 고려하지는 않음.
PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176),
]

def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
        
    return image

# 겹쳐진 클래스 고려한 시각화
def label2rgba(label):
    image_size = (512,512) + (4, )  # Add an alpha channel
    result_image = Image.new('RGBA', image_size[:-1], (0, 0, 0, 0))
    
    for i, class_label in enumerate(label):
        class_label = resize(class_label, (512,512))
        mask = class_label == 1
        image = np.zeros(image_size, dtype=np.uint8)
        image[mask] = PALETTE[i] + (128,)  # Add opacity
        result_image = Image.alpha_composite(result_image, Image.fromarray(image, 'RGBA'))
    
    return result_image


# 실제 정답인데 정답이 아니라고 한 것 (false negative)
def fn2rgb(true, pred):
    image_size = (512, 512) + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, pred_label in enumerate(pred):
        pred_label = resize(pred_label, (512,512))
        true_label = resize(true[i], (512,512))
        mistake = pred_label != true_label
        false_negative = mistake & (true_label == 1)
        image[false_negative] = PALETTE[i]
        
    return image
        

# 실제 정답이 아닌데 정답이라고 한 것 (false positive)
def fp2rgb(true, pred):
    image_size = (512, 512) + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, pred_label in enumerate(pred):
        pred_label = resize(pred_label, (512,512))
        true_label = resize(true[i], (512,512))
        mistake = pred_label != true_label
        false_positive = mistake & (true_label == 0)
        image[false_positive] = PALETTE[i]
        
    return image
    

# confusion matrix
def confusion_matrix():
    return None