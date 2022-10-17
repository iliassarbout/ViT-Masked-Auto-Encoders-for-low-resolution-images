from tqdm import tqdm
from PIL import Image
import numpy as np

def shape_filter(imgs_list):
    img_size = 64
    channels = 3
    ref_shape =(img_size,img_size,channels)
    imgs_good_shape = []
    for i in tqdm(range(len(imgs_list))): #img = Image.open(img_path)
        img = Image.open(imgs_list[i][0])
        img = img.resize((img_size, img_size))
        if np.array(img).shape==ref_shape:
            imgs_good_shape.append(imgs_list[i])
    print('Number of bad images : ', len(imgs_list)-len(imgs_good_shape), ' / ', len(imgs_list))
    return(imgs_good_shape)