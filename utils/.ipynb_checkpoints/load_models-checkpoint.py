import sys
import torch
from functools import partial
from torch import nn
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch

# setting path
sys.path.append('../')
from models_mae import MaskedAutoencoderViT

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
        checkpoint = torch.load(file_name, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print('Model loaded.')
    return(model)


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.backbone = base_model

        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def features(self,x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return(x)

    def forward(self, x):
        return self.backbone(x)
    
def load_simclr(file_name = 'checkpoint-49.pth'):
    resnet = torchvision.models.resnet18(pretrained=False, num_classes=128)
    #resnet(imgs_batch)

    model_simclr = ResNetSimCLR(resnet,128)
    #checkpoint = torch.load('checkpoint_0100.pth.tar', map_location=device)
    #state_dict = checkpoint['state_dict']
    #log = model.load_state_dict(state_dict, strict=False)
    #print(log.missing_keys)

    if file_name:
        checkpoint = torch.load(file_name, map_location='cpu')
        model_simclr.load_state_dict(checkpoint['model'])
        print('Model loaded.')

    return(model_simclr)