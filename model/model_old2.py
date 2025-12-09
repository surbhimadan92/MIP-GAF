import os
# os.environ['CUDA_ENABLE_DEVICES'] = '6'

import torch 
import torch.nn as nn
import torch.nn.functional as F
import CLIP.clip as clip 

from model.attn import PixelAttention

class ConvBNRelu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, G=32, use_relu=True):
        super(ConvBNRelu, self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.use_relu:
            self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=False)



class TRIS(nn.Module):
    def __init__(self, bert_tokenizer='clip', backbone='clip-RN50', max_query_len=20):
        super().__init__()

        self.clip = bert_tokenizer
        
        type = backbone.split('-')[-1]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load(type, device=device, jit=False, txt_length=max_query_len)
        # clip_model = clip_model.eval().float() 
        clip_model = clip_model.float() 
        self.backbone = clip_model 

        self.textdim = 512 

        ################################################################################################################
        if type == 'RN50':
            v_chans = [256, 512, 1024, 2048]  # 1024 or 2048, attention pool 
        elif type == 'RN101':
            v_chans = [256, 512, 1024, 2048]
        l_chans = self.textdim 

        self.attention2 = PixelAttention(visual_channel=v_chans[1], language_channel=l_chans)
        self.attention3 = PixelAttention(visual_channel=v_chans[2], language_channel=l_chans)
        self.attention4 = PixelAttention(visual_channel=v_chans[3], language_channel=l_chans)
        ################################################################################################################
        self.reduced_c1 = ConvBNRelu(v_chans[0], 64, kernel_size=3, stride=1, padding=1)
        self.reduced_c2 = ConvBNRelu(v_chans[1], 128, kernel_size=3, stride=1, padding=1)
        self.reduced_c3 = ConvBNRelu(v_chans[2], 256, kernel_size=3, stride=1, padding=1)
        self.reduced_c4 = ConvBNRelu(v_chans[3], 512, kernel_size=3, stride=1, padding=1)
        ##################
        self.final_seg1 = nn.Sequential(
            ConvBNRelu(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, bias=False))
        self.final_seg2 = nn.Sequential(
            ConvBNRelu(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, bias=False))
        self.final_seg3 = nn.Sequential(
            ConvBNRelu(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, bias=False))
        self.final_seg4 = nn.Sequential(
            ConvBNRelu(256, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, bias=False))
        # ##################
        self.output4 = ConvBNRelu(512, 256, kernel_size=3, stride=1, padding=1)
        self.output3 = ConvBNRelu(256, 128, kernel_size=3, stride=1, padding=1)
        self.output2 = ConvBNRelu(128, 64, kernel_size=3, stride=1, padding=1)
        self.output1 = ConvBNRelu(64, 32, kernel_size=3, stride=1, padding=1)   

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_bbox1 = nn.Linear(512, 256)
        self.fc_bbox2 = nn.Linear(256, 4)

    
    def trainable_parameters(self):
        # print(self.parameters())
        backbone = []
        head = []
        for k, v in self.named_parameters():
            if k.startswith('backbone') and 'positional_embedding' not in k:
                backbone.append(v)
            else:
                head.append(v)
        print('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
        return backbone, head 

    def forward(self, x, word_id):
        
        img_size = x.shape[2:]
        _,_,H,_ = x.size()
        
        word_embedding, hidden = self.backbone.encode_text(word_id) 
        
        # c1, c2, c3, c4 = self.backbone.encode_image(x)  
        c1, c2, c3, c4, attn_out = self.backbone.encode_image(x)  
        

        lan = word_embedding.permute(0, 2, 1)  # [N, T, C] -> [N, C, T]

        fuse_feats2 = self.attention2(c2, lan) + c2 
        fuse_feats3 = self.attention3(c3, lan) + c3 
        fuse_feats4 = self.attention4(c4, lan) + c4 

        dem1 = self.reduced_c1(c1)
        dem2 = self.reduced_c2(fuse_feats2)
        dem3 = self.reduced_c3(fuse_feats3)
        dem4 = self.reduced_c4(fuse_feats4)

        # print(f"dem1 shape: {dem1.shape}")
        # print(f"dem2 shape: {dem2.shape}")
        # print(f"dem3 shape: {dem3.shape}")
        # print(f"dem4 shape: {dem4.shape}")

        # seg_out4 = Upsample(self.output4(dem4), dem3.shape[2:])  # 512 -> 256 
        # seg_out3 = Upsample(self.output3(seg_out4 + dem3), dem2.shape[2:])  # [2, 128, 40, 40]
        # seg_out2 = Upsample(self.output2(seg_out3 + dem2), dem1.shape[2:])  # [2, 64, 80, 80]
        # seg_out1 = self.output1(seg_out2 + dem1)

        # print(f"seg_out4 shape: {seg_out4.shape}")
        # print(f"seg_out3 shape: {seg_out3.shape}")
        # print(f"seg_out2 shape: {seg_out2.shape}")
        # print(f"seg_out1 shape: {seg_out1.shape}")
        
        
        pooled_output = self.global_avg_pool(dem4)
        # print(f"After pooling: {pooled_output.shape}")
        pooled_output = pooled_output.view(pooled_output.size(0), -1)
        
        pooled_output = F.relu(self.fc_bbox1(pooled_output))
        bbox_out = self.fc_bbox2(pooled_output)

        return bbox_out

# def criterion(seg_final_out1, target):
#     target = target.float() 
#     return F.binary_cross_entropy_with_logits(input=seg_final_out1, target=target, reduction='mean')
def criterion(bbox_out, target):
    target = target.float()
    return nn.MSELoss()(bbox_out, target)

