import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch 
import os
import torch.nn.functional as F
from model.model_stage2_refine_v5 import TRIS, criterion 
# from model.model import TRIS, criterion
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

import matplotlib.pyplot as plt

img_size = 320

transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class MIPDataset(Dataset):

    def __init__(self, root='./Group_Emotion_Recognition', split='Train', transform=transform, emotion='Negative', desc_type='short', img_size=img_size, max_length=20):
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
        try:
            desc = desc.replace(',', '')
        except Exception as e:
            print(img_name)
        word_ids = []
        # word_ids = []
        # split_text = desc.split(',')
        # tokenizer = clip.tokenize
        # for text in split_text:
        #     word_id = tokenizer(text).squeeze(0)[:self.max_length]
        #     word_ids.append(word_id.unsqueeze(0))
        # word_ids = torch.cat(word_ids, dim=-1)
        # word_id = word_id.view(-1)
        split_text = desc.split('.')
        tokenizer = clip.tokenize
        
        for text in split_text:
            word_id = tokenizer(text[:77]).squeeze(0)[:self.max_length]
            word_ids.append(word_id.unsqueeze(0))

    
        ## Fetching bbox
        with open(f"{self.root}/GAF3.0_JSON/{self.split}/{self.emotion}/{img_name.split('.')[0]}_bbox.json", 'r') as f:
            bbox_json = json.load(f)
            x1, y1, x2, y2  = bbox_json['BBox_Normalized']
        x1, y1, x2, y2 = int(x1*self.img_size), int(y1*self.img_size), int(x2*self.img_size), int(y2*self.img_size)

        target = torch.zeros(size=(img_size, img_size))
        target[y1:y2, x1:x2] = 1
        

        return img, word_ids, target

def collate_fn(batch):
    img, word_ids, target = zip(*batch)

    img = torch.stack(img)
    target = torch.stack(target)

    max_len = max(len(word_id) for word_id in word_ids)

    padded_word_ids = []
    for word_id in word_ids:
        while len(word_id)!=max_len:
            word_id.append(torch.zeros(size=(1, 20)))
        padded_word_ids.append(torch.stack(word_id))

    padded_word_ids = torch.stack(padded_word_ids)

    return img, padded_word_ids, target