import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def images_to_sprite(data):
        #create sprite image and necessary padding
        if len(data.shape) == 3:
            data = np.tile(data[...,np.newaxis], (1,1,1,3))
        data = data.astype(np.float32)
        min = np.min(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
        max = np.max(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)

        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, 0),
                (0, 0)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant',
                constant_values=0)

        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        data = (data * 255).astype(np.uint8)
        return data
        
        

class DATA(Dataset): #Allows to extract images as tensor or PIL images.
    def __init__(self, img_list, transform = None, mean = None,std = None,img_size = 64):
        self.img_list = img_list
        self.transform = None
        self.mean = mean
        self.std = std
        self.img_size = img_size
        self.normalize = None
        if(self.mean is not None and self.std is not None):
            self.normalize = transforms.Normalize(mean=self.mean,
                         std=self.std)
        self.img_size = img_size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx, pil=False):
        img_path = self.img_list[idx][0]
        img = Image.open(img_path)
        img = img.resize((self.img_size, self.img_size))
        if pil:
            return img,self.img_list[idx][1]
        img = np.array(img)/255.              
        img = torch.Tensor(img) 
        img = torch.einsum('hwc->chw', img) #Switch channel position
        if self.normalize is not None:
            img = self.normalize(img)
        #img = tf.convert_to_tensor(img)
        return img,self.img_list[idx][1]
    
    
class extractor():
    
    def __init__(self, model,dataset):
        self.model = model
        self.dataset = dataset
 
    def get_embed(self,id,mask_ratio=0.75,grid_id=0):
        img = self.dataset.__getitem__(id)[0].unsqueeze(dim=0)
        img = img.to('cuda')
        with torch.no_grad():
            f0,_,_,_ = self.model.forward_encoder(img,mask_ratio = mask_ratio,grid_idx = grid_id)
        f0 = f0[:, 1:, :] 
        return(f0.flatten())

    def get_img(self,id):
        img = self.dataset.__getitem__(id,pil=True)[0]
        return(img)

    def get_label(self,id):
        return(self.dataset.img_list[id][1])