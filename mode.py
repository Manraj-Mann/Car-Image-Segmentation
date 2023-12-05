import os
import random
import zipfile

import numpy as np
import torch
import torchvision.models as models
import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

class BaseClass(nn.Module):
    def training_step(self, batch):
        inputs, targets = batch        
        preds = self(inputs)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(preds, targets)
        return loss
    
    def validation_step(self, batch, score_fn):
        inputs, targets = batch
        preds = self(inputs)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(preds, targets)
        score = score_fn(preds, targets)
        return {'val_loss': loss.detach(), 'val_score': score}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = sum(batch_losses)/len(batch_losses)
        
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = sum(batch_scores)/len(batch_scores)
        
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score}
    
    def epoch_end(self, epoch, nEpochs, results):
        print("Epoch: [{}/{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score:{:.4f}".format(
                        epoch+1, nEpochs, results['train_loss'], results['val_loss'], results['val_score']))


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size = 3,stride = 1, padding = 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)

def copy_and_crop(down_layer, up_layer):
    b, ch, h, w = up_layer.shape
    crop = T.CenterCrop((h, w))(down_layer)
    return crop
    
class UNet(BaseClass):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList([
            conv_block(in_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512)
        ])
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottle_neck = conv_block(512, 1024)
        
        self.up_samples = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])
        
        self.decoder = nn.ModuleList([
            conv_block(1024, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64)
        ])
        
        self.final_layer = nn.Conv2d(64, out_channels, 1, 1)
        
    def forward(self, x):
        skip_connections = []
        
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottle_neck(x)
        
        for ind, layer in enumerate(self.decoder):
            x = self.up_samples[ind](x)
            y = copy_and_crop(skip_connections.pop(), x)
            x = layer(torch.cat([y, x], dim=1))
        
        x = self.final_layer(x)
        
        return x
    
class MyModal:

    val_transforms = A.Compose([
            A.Resize(height=256, width=256),
            ToTensorV2()
        ])
    def get_model(self):
        model = UNet(3, 1)
        model_path = "unet_segmentation.pth"
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        return model

    def process_custom_image(self , image):

        img = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        img_t = self.val_transforms(image=img)['image'].to('cpu')
        model = self.get_model()
        logits = model(img_t.unsqueeze(0)).detach().cpu()
        preds = F.sigmoid(logits)
        preds = (preds>0.5).float().detach().cpu()
        mask_image = preds[0].permute(1, 2, 0).numpy()
        generated_image = ( mask_image * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
        
        # Convert the cv2 image (NumPy array) to a PIL image
        pil_image = Image.fromarray(generated_image)    
        # pil_image.save("masked_image.png")
        
        
        resized_image = image.resize(pil_image.size)
        # resized_image.save("original_image.png")
        
        print(pil_image.size , resized_image.size)
        
        mask = np.array(pil_image) 
        org = np.array(resized_image)
        print(mask.shape)
        print(org.shape)
        
        for i in range(len(org)):
            for j in range(len(org[i])):
                if mask[i][j][0] != 0:
                    org[i][j][0] = 250


        output_image = Image.fromarray(org)
        # output_image.save("output_image.jpg")

        return [resized_image , pil_image , output_image]
