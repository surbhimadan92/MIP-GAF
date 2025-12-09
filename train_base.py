import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch 
import os
import torch.nn.functional as F
# from model.model_stage2 import TRIS, criterion 
from model.model import TRIS, criterion
# from model.model_old2 import TRIS, criterion

import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler
from dataset.ReferDataset import ReferDataset
from dataset.transform import get_transform
from args import get_parser
import random
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, sampler
from utils.poly_lr_decay import PolynomialLRDecay
from utils.util import AverageMeter, load_checkpoint,reduce_tensor, save_checkpoint, load_pretrained_checkpoint
import time 
from logger import create_logger
import datetime
from utils.util import compute_mask_IU 
import cv2 
import numpy as np 
import pdb 
import CLIP.clip as clip 

from ema_pytorch import EMA
from validate import validate 
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs.log', level=logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device=='cuda':
    torch.cuda.empty_cache()

print(f"Running on {device}")


## Defining parameters
img_size = 320
batch_size = 32

epochs = 200
# lr = 0.00005
lr = 0.001
lr_multi = 0.1
weight_decay = 0.01

warm_up = 20

pretrained_path = './weights/stage2_refcocog_google.pth'


## Defining transform
transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class MIPDataset(Dataset):

    def __init__(self, root='../Group_Emotion_Recognition', split='Train', transform=transform, emotion='Negative', desc_type='short', img_size=img_size, max_length=20):
        super().__init__()

        assert desc_type in ['short', 'long'], "desc_type must be either short or long"

        self.split = split
        self.emotion = emotion
        self.root = root
        self.max_length = max_length

        ## Image directory
        self.img_dir = f"{root}/GAF3.0_Original/{split}/{emotion}"

        ## Image size
        self.img_size = img_size

        ## Short description csv
        self.short_desc = pd.read_csv(f"{root}/Image_short_description/{split.lower()}_{emotion.lower()}.csv")

        ## Long description csv
        self.long_desc = pd.read_csv(f"{root}/Image_Description/{split}_{emotion}_combined.csv")

        ## Image transforms
        self.transform = transform

        self.img_list = sorted(self.short_desc['image_name'].unique())

        ## BBox locations
        self.bbox_dir = f"{root}/GAF3.0_JSON/{split}/{emotion}"

        ## Setting image name as index
        self.short_desc = self.short_desc.set_index('image_name')
        self.long_desc = self.long_desc.set_index('image_name')

        ## Description type to use
        self.desc_type = desc_type


    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        ## Image path
        img_path = f"{self.img_dir}/{img_name}"
        img = cv2.imread(img_path)

        img = Image.fromarray(img)

        img = self.transform(img)

        ## Fetching description
        if self.desc_type=='short':
            desc = self.short_desc.loc[img_name, 'description']
        else:
            desc = self.long_desc.loc[img_name, 'description']

        desc = desc.replace(',', '')
        word_ids = []
        split_text = desc.split(',')
        tokenizer = clip.tokenize
        for text in split_text:
            word_id = tokenizer(text).squeeze(0)[:self.max_length]
            word_ids.append(word_id.unsqueeze(0))
        word_ids = torch.cat(word_ids, dim=-1)
        word_id = word_id.view(-1)
    
        ## Fetching bbox
        with open(f"{self.root}/GAF3.0_JSON/{self.split}/{self.emotion}/{img_name.split('.')[0]}_bbox.json", 'r') as f:
            bbox_json = json.load(f)
            bbox = bbox_json['BBox_Normalized']
        bbox = (np.array(bbox)*self.img_size).astype(int)
        # bbox = np.array(bbox)

        return img, word_ids, torch.from_numpy(bbox)
    

def cancel_warmup(model):
    for name, params in model.named_parameters():
        params.requires_grad = True

if __name__=='__main__':

    trainset_pos, trainset_neg, trainset_neu = MIPDataset(emotion='Positive'), MIPDataset(emotion='Negative'), MIPDataset(emotion='Neutral')
    valset_pos, valset_neg, valset_neu = MIPDataset(split='Validation', emotion='Positive'), MIPDataset(split='Validation', emotion='Negative'), MIPDataset(split='Validation', emotion='Neutral')

    trainset = ConcatDataset([trainset_pos, trainset_neg, trainset_neu])
    valset = ConcatDataset([valset_pos, valset_neg, valset_neu])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    model = TRIS().to(device)
    model.load_state_dict(torch.load(pretrained_path)['model'], strict=False)
    model.train()

    param_groups = model.trainable_parameters()

    optimizer = AdamW([
                    {'params': param_groups[0], 'lr': lr * lr_multi, 'weight_decay': weight_decay},
                    {'params': param_groups[1], 'lr': lr, 'weight_decay': weight_decay},
                ], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                        lambda x: (1 - x / (len(train_loader) * epochs)) ** 0.9)
    
    for name, params in model.named_parameters():
        if 'bbox' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False

    min_loss = np.inf
    for epoch in range(epochs):

        model.train()
        if epoch+1==warm_up:
            cancel_warmup(model)

        losses = []
        for img, word_ids, bbox in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            img, word_ids, bbox = img.to(device), word_ids.to(device), bbox.to(device)
            bbox_out = model(img, word_ids.squeeze(1))
            loss = criterion(bbox_out, bbox)


            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        mean_loss = np.mean(losses)
        print(f"Training - Epoch: {epoch+1} - Mean Loss: {mean_loss}")
        logging.info(f"Training - Epoch: {epoch+1} - Mean Loss: {mean_loss}")

        model.eval()
        losses = []
        
        for img, word_ids, bbox in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{epochs}"):
            with torch.no_grad():
                img, word_ids, bbox = img.to(device), word_ids.to(device), bbox.to(device)
                bbox_out = model(img, word_ids.squeeze(1))
                # print(bbox_out)
                loss = criterion(bbox_out, bbox)
                

                losses.append(loss.item())

        mean_loss = np.mean(losses)
        if mean_loss<min_loss:
            min_loss = mean_loss
            torch.save(model.state_dict(), './ckpt/model_v3_001.pt')
        print(f"Validating - Epoch: {epoch+1} - Mean Loss: {mean_loss}")
        logging.info(f"Validating - Epoch: {epoch+1} - Mean Loss: {mean_loss}")
