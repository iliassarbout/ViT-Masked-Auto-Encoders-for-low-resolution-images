import sys
import torch
from functools import partial
from torch import nn

# setting path
sys.path.append('../')


from models_mae import MaskedAutoencoderViT
#import torch

def load_mae(n_heads=16,patrch_size=8,img_size=64,file_name='checkpoint-99.pth'):

    n_heads = 16
    patch_size = 8
    img_size = 64
    num_patch = int((img_size/patch_size)**2)
    model = MaskedAutoencoderViT(
            img_size=img_size,patch_size=patch_size, embed_dim=240, depth=10, num_heads=12,
            decoder_embed_dim=160, decoder_depth=6, decoder_num_heads=n_heads,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    if file_name:
        to_load = True
    if to_load:
        checkpoint = torch.load(file_name, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print('Model loaded.')
    return(model)