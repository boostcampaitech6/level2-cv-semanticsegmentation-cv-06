# python native
import os
import datetime

# external library
import numpy as np
from tqdm.auto import tqdm
import albumentations as A
from PIL import Image
import argparse

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms.functional import to_pil_image

from dataset import XRayDataset
from utils import set_seed, label2rgb, label2rgba, fp2rgb, fn2rgb
from loss import focal_loss, dice_loss, calc_loss
import wandb

import warnings

warnings.filterwarnings("ignore")

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

# dice
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

# 모델 저장
def save_model(model, save_dir, file_name='best.pt'):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, file_name)
    torch.save(model, output_path)
    
def visualize_and_log_wandb(results, epoch):
    for result in results:
        for output, mask, image, image_path in result:

            output_np = output.numpy()
            mask_np = mask.numpy()
            image_np = image.numpy().transpose(1,2,0)
            file_name = '/'.join(image_path.split('/')[-2:])

            output_rgba = label2rgba(output_np)
            mask_rgba = label2rgba(mask_np)
            fp_rgb = fp2rgb(mask_np, output_np)
            fn_rgb = fn2rgb(mask_np, output_np)

            #Log images to wandb
            wandb.log({f"images on {epoch+1} epochs": 
                    [wandb.Image(image_np, caption=f"GT Image \n {file_name}"),
                    wandb.Image(mask_rgba, caption="GT Mask"),
                    wandb.Image(output_rgba, caption="Model Prediction"),
                    wandb.Image(to_pil_image(fp_rgb), caption="false positive"),
                    wandb.Image(to_pil_image(fn_rgb), caption="false negative")]
                    })
    
# validation
def validation(epoch, model, val_loader, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    results = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, masks, image_paths) in tqdm(enumerate(val_loader), total=len(val_loader)):
            images, masks = images.cuda(), masks.cuda()          
            model = model.cuda()    
            
            outputs = model(images)['out']
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr)
            
            dice = dice_coef(masks, outputs)
            dices.append(dice)
            
            result = [(o, m, i, p) for o, m, i, p in zip(outputs.detach().cpu(), masks.detach().cpu(), images.detach().cpu(), image_paths)]
            results.append(result)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice, results, dice_str

def train(args, model, train_loader, val_loader, criterion, optimizer, scheduler):
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.
    serial = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for epoch in range(args.num_epochs):
        model.train()

        for step, (images, masks, _) in enumerate(train_loader):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()    
            model = model.cuda()        
            outputs = model(images)['out']
            
            # loss를 계산합니다.
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{args.num_epochs}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({'Loss': round(loss.item(),4), 'learning rate':current_lr})
             
        scheduler.step()
            
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        # validation 함수는 위에 정의됨
        if (epoch + 1) % args.val_interval == 0:
            dice, results, dice_log = validation(epoch + 1, model, val_loader)
            
            wandb.log({'dice score': dice})
            
            print("Wandb image logging ...")
            visualize_and_log_wandb(results, epoch)
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {args.save_dir}")
                best_dice = dice
                save_model(model, os.path.join(args.save_dir, serial), file_name=f'{epoch+1}.pt')
                save_model(model, os.path.join(args.save_dir, serial), file_name='best.pt')
                
                dice_per_class_log = os.path.join(args.save_dir, serial, 'dice_per_class.txt')
                with open(dice_per_class_log, 'a', encoding='utf-8') as file:
                    file.write("="*10+f" validation_{epoch+1} "+"="*10 +"\n")
                    file.write(dice_log)
                    file.write("\n")
                
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--image_root', help='train image file path', default='/data/ephemeral/home/data/train/DCM')
    parser.add_argument('--label_root', help='label image file path', default='/data/ephemeral/home/data/train/outputs_json')
    parser.add_argument('--save_dir', help='save pt dir path', default='/data/ephemeral/home/level2-cv-semanticsegmentation-cv-06/pts')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--val_interval', type=int, default=20)
    args = parser.parse_args()

    return args

def main(args):
    
    # seed 666
    set_seed()
    
    # transform 정의
    tf = A.Resize(512, 512)
    
    train_dataset = XRayDataset(args.image_root, args.label_root, is_train=True, transforms=tf)
    valid_dataset = XRayDataset(args.image_root, args.label_root, is_train=False, transforms=tf)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=8,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    #model = models.segmentation.fcn_resnet101(pretrained=True)
    #model.classifier[4] = nn.Conv2d(512, len(CLASSES), kernel_size=1)
    
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.classifier[-1] = torch.nn.Conv2d(256, len(CLASSES), kernel_size=(1, 1))
    
    criterion = calc_loss
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.num_epochs//2], gamma=0.1)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs//4, eta_min=1e-5)
    
    train(args, model, train_loader, valid_loader, criterion, optimizer, scheduler)

if __name__ == '__main__':
    args = parse_args()
    
    wandb.init(project="CV06_Segmantation",
               #entity="innovation-vision-tech",
               name=f"deeplabv3_resnet101_{args.num_epochs}e_calcloss(dice3,bce1)_adam_MultiStepLR",
               notes="",
               config={
                    "batch_size": args.batch_size,
                    "Learning_rate": args.lr,
                    "Epochs": args.num_epochs,
                })
    
    start_time = datetime.datetime.now()
    main(args)
    end_time = datetime.datetime.now()
    
    elapsed_time = end_time - start_time
    print(f"총 학습 시간: {str(elapsed_time)}")