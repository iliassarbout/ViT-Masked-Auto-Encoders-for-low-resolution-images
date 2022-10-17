import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob 
import os 
from tqdm import tqdm

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
    
class DATA_2(Dataset): #this one can resize, take only labels up to a certain one, extract pil
    def __init__(self, img_list, labels_list, transform = None, mean = None,std = None,img_size = 64,max_label = None):
        self.img_list = img_list
        self.labels_list = labels_list
        if max_label is not None:
          idx = np.where(np.array(self.labels_list)<=max_label)[0]
          self.img_list = list(np.take(self.img_list,idx))
          self.labels_list = list(np.take(self.labels_list,idx))
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
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        img = img.resize((self.img_size, self.img_size))
        if pil:
          return img,self.img_list[idx][1]
        img = np.array(img)/255.              
        img = torch.Tensor(img) 
        img = torch.einsum('hwc->chw', img) #switch channel position
        if self.normalize is not None:
            img = self.normalize(img)
        #img = tf.convert_to_tensor(img)
        return img,self.labels_list[idx]


    
class extractor():
    
    def __init__(self, model,dataset,device):
        self.model = model.to
        self.dataset = dataset
        self.device = device
     
    def get_embed(self,id,mask_ratio=0.75,grid_id=0):
        img = self.dataset.__getitem__(id)[0].unsqueeze(dim=0)
        img = img.to(self.device)
        with torch.no_grad():
            f0,_,_,_ = self.model().forward_encoder(img,mask_ratio = mask_ratio,grid_idx = grid_id)
        f0 = f0[:, 1:, :] 
        return(f0.flatten())
    
    def get_embed_simclr(self,id,cuda=False):
          img = self.dataset.__getitem__(id)[0].unsqueeze(dim=0)
          img = img.to(self.device)
          f0 = self.model().features(img)[0] #squeeze
          return(f0) #squeeze
    
    def get_img(self,id):
        img = self.dataset.__getitem__(id,pil=True)[0]
        return(img)

    def get_label(self,id):
        return(self.dataset.img_list[id][1])
    

def path_labels_extractor(path,img_size = 64,channels=3,ext='jpg'):
    ref_shape =(img_size,img_size,channels)

    classes = {}
    img_list = []
    img_labels = []
    classes_list = glob.glob(os.path.join(path,'*'))
    path_length = len(path)+1

    k = 0
    n_bad = 0
    for k in tqdm(range(len(classes_list))):
      classes[str(k)] = classes_list[k][path_length:]  
      class_imgs = glob.glob(os.path.join(classes_list[k], '*.'+ext) )
      for j in class_imgs:
        img = Image.open(j)
        img = img.resize((img_size, img_size))
        if np.array(img).shape==ref_shape:
          img_list.append(j)
          img_labels.append(k)
        else:
            n_bad+=1
    return(img_list,img_labels,classes,n_bad)
 