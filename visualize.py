from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
inet_mean = [0.485, 0.456, 0.406]
inet_std = [0.229, 0.224, 0.225]

def to_print(image,std=inet_std,mean=inet_mean, title='',show=True):
    if not torch.is_tensor(mean):
      mean = torch.tensor(mean)
    if not torch.is_tensor(std):
      std = torch.tensor(std)
    #print(image.shape)
    if image.shape[2] != 3:
      image = torch.einsum('chw->hwc', image)
    assert image.shape[2] == 3
    to_print = torch.clip((image * std + mean) * 255, 0, 255).int()
    if show:
      plt.imshow(to_print)
      plt.title(title, fontsize=16)
      plt.axis('off')
      return()
    return(to_print)

def visualize_mae(img,y,mask,img_size=224,patch_size=16):
  l = int(img_size/patch_size) #num patch per line
  num_patch = int(l**2)
  img = torch.einsum('chw->hwc', img)
  r = torch.ones(3,img_size,img_size,3) #last number is channels
  r[0] = img
  for i in range(num_patch):
      x1 = patch_size*(i%l)
      y1 = x1 + patch_size
      x2 = patch_size*(i//l)
      y2 = x2+patch_size
      tmp = int(mask[i])
      patch_recons = y[i]
      r[2,x2:y2,x1:y1,:] = patch_recons.view(patch_size,patch_size,3)
      if(tmp==0): #if unmasked
        r[1,x2:y2,x1:y1,:] = img[x2:y2,x1:y1,:].view(patch_size,patch_size,3)
      elif(tmp==1):
        r[1,x2:y2,x1:y1,:] = torch.zeros((patch_size,patch_size,3))      
  return(r.view(3,img_size,img_size,3))

def visualize_mae_3pictures(img,y,mask,std,mean,img_size=224,patch_size=16):
  if not torch.is_tensor(mean):
      mean = torch.tensor(mean)
  if not torch.is_tensor(std):
      std = torch.tensor(std)
  original,reconsplus,recons = visualize_mae(img,y,mask,img_size,patch_size).detach()
  figure = plt.figure(figsize=(16, 16))
  rows = 3

  figure.add_subplot(1, 3, 1)
  plt.title('Original')
  plt.axis("off")
  plt.imshow(torch.clip((original * std + mean) * 255, 0, 255).int()) #could need to use img.squeeze() to remove any dimension equal to 1 on the input

  figure.add_subplot(1, 3, 3)
  plt.title('Reconstructed')
  plt.axis("off")
  plt.imshow(torch.clip((recons * std + mean) * 255, 0, 255).int()) #could need to use img.squeeze() to remove any dimension equal to 1 on the input

  figure.add_subplot(1, 3, 2)
  plt.title('Input')
  plt.axis("off")
  plt.imshow(torch.clip((reconsplus * std + mean) * 255, 0, 255).int()) #could need to use img.squeeze() to remove any dimension equal to 1 on the input

  plt.show()

def visualize_neighbors(features,data,id,k,dist='euclidian'):
  img = features[id].unsqueeze(dim=0)
  if dist=='euclidian':
    dists = torch.norm(features - img,dim=1)
    desc = False
  if dist=='cosine':
    dists = F.cosine_similarity(img,features)
    desc = True
  top_k = torch.argsort(dists, dim=- 1, descending=desc)[1:k+1]
  
  
  figure = plt.figure(figsize=(16, 16))
  rows = k+1

  figure.add_subplot(1, rows, 1)
  plt.title('Original')
  plt.axis("off")
  original = data.__getitem__(id)
  original = torch.einsum('chw->hwc', original)
  plt.imshow(torch.clip((original * std + mean) * 255, 0, 255).int()) #could need to use img.squeeze() to remove any dimension equal to 1 on the input

  for i in range(k):
    figure.add_subplot(1, rows, i+2)
    img = data.__getitem__(top_k[i])
    img = torch.einsum('chw->hwc', img)
    title = 'Neighbor '+str(i+1)
    plt.title(title)
    plt.axis("off")
    plt.imshow(torch.clip((img * std + mean) * 255, 0, 255).int()) #could need to use img.squeeze() to remove any dimension equal to 1 on the input

  plt.show()