# python native
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from dataset import XRayInferenceDataset
import argparse

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]
CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
IND2CLASS = {v: k for k, v in CLASS2IND.items()}

# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# RLE로 인코딩된 결과를 mask map으로 복원합니다.
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)['out']
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--image_root', help='train image file path', default='/data/ephemeral/home/data/test/DCM')
    parser.add_argument('--serial', help='serial of save pt path', default='20240000_000000')
    parser.add_argument('--pt', type=str, default='best.pt')
    parser.add_argument('--res_dir', help='save pth dir path and serial', default='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-06/results')
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    return args


def main(args):
    SAVE_DIR = '/data/ephemeral/home/level2-cv-semanticsegmentation-cv-06/pts/'
    save_dir = os.path.join(SAVE_DIR, args.serial)
    model = torch.load(os.path.join(save_dir, args.pt))
    
    tf = A.Resize(512, 512)
    test_dataset = XRayInferenceDataset(args.image_root, transforms=tf)

    test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=2,
    shuffle=False,
    num_workers=2,
    drop_last=False
    )
    
    rles, filename_and_class = test(model, test_loader)
    
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    
    df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
    })
    
    result_dir = os.path.join(args.res_dir, args.serial)
    os.makedirs(result_dir, exist_ok=True)
    
    df.to_csv(os.path.join(result_dir, "output.csv"), index=False)
    
    if args.verbose:
        res_img_dir = os.path.join(result_dir, 'pred_img')
        os.makedirs(res_img_dir, exist_ok=True)
        
        preds = []
        for rle in rles[:len(CLASSES)]:
            pred = decode_rle_to_mask(rle, height=2048, width=2048)
            preds.append(pred)

        preds = np.stack(preds, 0)

if __name__ == '__main__':
    args = parse_args()
    
    main(args)