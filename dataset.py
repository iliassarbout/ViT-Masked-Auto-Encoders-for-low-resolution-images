import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

class DATA(Dataset):
    def __init__(self, img_list, transform = None, mean = None,std = None):
        self.img_list = img_list
        self.transform = transform
        self.mean = mean
        self.std = std

        if(self.mean is not None and self.std is not None):
            self.normalize = transforms.Normalize(mean=self.mean,
                         std=self.std)
        else:
            self.normalize = None
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = torch.load(img_path)
        
        #img = img.unsqueeze(diml=0) #add a dimension (same as img = img[None,:]) #not necssary as dataloder will create batch dimension
        if self.normalize is not None:
            img = self.normalize(img)
        if self.transform is not None:
            img = self.transform(img)
        #img = torch.einsum('hwc->chw', img) #switch channel position

        return img,idx

