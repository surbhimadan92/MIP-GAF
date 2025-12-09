import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import torch 
import os
import torch.nn.functional as F
from model.model_stage2_refine_v5 import MIPCLIP, criterion 
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

from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm

from dataset.mip import MIPDataset, collate_fn

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs.log', level=logging.INFO)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device=='cuda':
    torch.cuda.empty_cache()

import matplotlib.pyplot as plt
print(f"Running on {device}")

def sigmoid_mse_loss(input_logits, target_logits):
    """Takes sigmoid on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_sigmoid = F.sigmoid(input_logits)
    target_sigmoid = F.sigmoid(target_logits)

    return F.mse_loss(input_sigmoid, target_sigmoid, reduction='mean')

def cancel_warmup(model):
    for name, params in model.named_parameters():
        params.requires_grad = True

if __name__=='__main__':
    img_size = 320

    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    trainset_pos, trainset_neg, trainset_neu = MIPDataset(emotion='Positive', desc_type='long'), MIPDataset(emotion='Negative', desc_type='long'), MIPDataset(emotion='Neutral', desc_type='long')
    valset_pos, valset_neg, valset_neu = MIPDataset(split='Validation', emotion='Positive', desc_type='long'), MIPDataset(split='Validation', emotion='Negative', desc_type='long'), MIPDataset(split='Validation', emotion='Neutral', desc_type='long')

    trainset = ConcatDataset([trainset_pos, trainset_neg, trainset_neu])
    valset = ConcatDataset([valset_pos, valset_neg, valset_neu])

    batch_size = 8

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    pretrained_path = './weights/stage2_refcocog_umd.pth'

    model = MIPCLIP().to(device)
    model.load_state_dict(torch.load(pretrained_path)['model'], strict=False)
    model.train()

    param_groups = model.trainable_parameters()

    epochs = 200
    lr = 0.00005
    # lr = 0.001
    lr_multi = 0.1
    weight_decay = 0.01

    optimizer = AdamW([
                    {'params': param_groups[0], 'lr': lr * lr_multi, 'weight_decay': weight_decay},
                    {'params': param_groups[1], 'lr': lr, 'weight_decay': weight_decay},
                ], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                        lambda x: (1 - x / (len(train_loader) * epochs)) ** 0.9)
    
    consistency_criterion = sigmoid_mse_loss

    model.backbone.requires_grad = False

    sav_dir = './ckpts'
    model_name = 'v5_005_warm_umd.pt'

    warm_up = 3

    for epoch in range(epochs):
    
        model.train()
        if epoch+1==warm_up:
            cancel_warmup(model)
            
        for img, word_ids, target in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            num_sentences = word_ids.shape[1]
            
            img, word_ids, target = img.to(device), word_ids.to(device).int(), target.to(device)

            out1 = []
            out2 = []
            out3 = []
            out4 = []
            for i in range(num_sentences):
                word_id = word_ids[:, i, :, :]

                
                output1, output2, output3, output4 = model(img, word_id.squeeze(1))

                out1.append(output1)
                out2.append(output2)
                out3.append(output3)
                out4.append(output4)

            out1 = torch.cat(out1, dim=1)
            out2 = torch.cat(out2, dim=1)
            out3 = torch.cat(out3, dim=1)
            out4 = torch.cat(out4, dim=1)

            out1 = torch.mean(out1, dim=1)
            out2 = torch.mean(out2, dim=1)
            out3 = torch.mean(out3, dim=1)
            out4 = torch.mean(out4, dim=1)

            l1 = criterion(out1, target)
            l2 = criterion(out2, target)
            l3 = criterion(out3, target)
            l4 = criterion(out4, target)

            loss = l1 + l2 + l3 + l4
            
            loss.backward()
            optimizer.step()

        model.eval()
        losses = []
        max_loss = np.inf
        for img, word_ids, target in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{epochs}"):
            num_sentences = word_ids.shape[1]
            with torch.no_grad():

                img, word_ids, target = img.to(device), word_ids.to(device).int(), target.to(device)
                out1 = []
                out2 = []
                out3 = []
                out4 = []
                
                for i in range(num_sentences):
                    word_id = word_ids[:, i, :, :]

                    
                    output1 = model(img, word_id.squeeze(1))

                    out1.append(output1)

                out1 = torch.cat(out1, dim=1)

                out1 = torch.mean(out1, dim=1)

                l1 = criterion(out1, target)
                
                losses.extend(l1.flatten().tolist())
        
        mean_loss = np.mean(losses)
        print(f"Mean Validation Loss: {mean_loss}")
        logging.info(f"Mean Validation Loss: {mean_loss}")
        if mean_loss<max_loss:
            max_loss = mean_loss

            torch.save(model.state_dict(), f"{sav_dir}/{model_name}")
