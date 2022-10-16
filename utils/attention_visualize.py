import torch
from collections import defaultdict

#utils for colors
def patch_gen(color,size): #returns patch of shape channels,size,size with color as list as input
  return(torch.Tensor(color).reshape(3,1).repeat(1,size).repeat(1,size).reshape(3,size,size))


def patch_gen_influencer(color,size,full=False): #returns patch of shape channels,size,size with color as list as input
  if full:
    return(torch.Tensor(color).reshape(3,1).repeat(1,size).repeat(1,size).reshape(3,size,size))
  patch = torch.Tensor([255,255,255]).reshape(3,1).repeat(1,size).repeat(1,size).reshape(3,size,size)
  a = int(size/4)
  b= int(3*size/4)
  patch[:,a:b,a:b] = torch.Tensor(color).reshape(3,1).repeat(1,size//2).repeat(1,size//2).reshape(3,size//2,size//2)
  return(patch)

def create_color_dict():
  colors = defaultdict(list)
  colors[0] = [255,0,255]
  colors[1] = [255,0,0]
  colors[2] = [128,0,0]
  colors[3] = [255,255,0]
  colors[4] = [128,128,0]
  colors[5] = [0,128,0]
  colors[6] = [0,255,0]
  colors[7] = [0,128,128]
  colors[8] = [0,255,255]
  colors[9] = [0,0,255]
  colors[10] = [128,0,128]
  colors[11] = [0,0,128]
  colors[12] = [192,192,192]
  colors[13] = [128,128,128]
  colors[14] = [200,127,0]
  colors[15] = [240,220,200]
  return(colors)

def keys_to_col(keys,colors):
  keys_col = defaultdict(list)
  for j in range(len(keys)):
    keys_col[keys[j]] = colors[j] 
  return(keys_col)


def draw_att(list_of_influence,colors,img_size=64,patch_size=8,channels=3,mix=True,field=False):
  l = int(img_size/patch_size) #num patch per line
  num_patch = int(l**2)
  count = {} #we count the number of time a patch is influenced to normalize and ensure the mix of colors is between 0 and 255
  img = torch.zeros(channels,img_size,img_size)
  keys = list(list_of_influence.keys())
  key_col = keys_to_col(keys,colors)

  if mix: #mix over most importants influencer, with respect to attention value
    for j in range(len(keys)):
      #drawing influencer path
      key = keys[j] 
      x1 = patch_size*(key%l)
      y1 = x1 + patch_size
      x2 = patch_size*(key//l)
      y2 = x2+patch_size
      img[:,x2:y2,x1:y1] = patch_gen_influencer(colors[j],patch_size)
      for patch_to_influence,attention in torch.Tensor(list_of_influence[key]): #adding influencer color's patch for each influenced patch
        patch_to_influence = int(patch_to_influence)
        if patch_to_influence not in keys:
          x1 = patch_size*(patch_to_influence%l)
          y1 = x1 + patch_size
          x2 = patch_size*(patch_to_influence//l)
          y2 = x2+patch_size
          img[:,x2:y2,x1:y1] = img[:,x2:y2,x1:y1] + attention*patch_gen(key_col[key],patch_size)
          count[patch_to_influence] = attention if patch_to_influence not in count.keys() else count[patch_to_influence]+attention

    #normalize influenced patchs
    for patch_influenced in count.keys():
      if patch_influenced not in keys: #line should be useless (to test)
        patch_influenced = int(patch_influenced)
        x1 = patch_size*(patch_influenced%l)
        y1 = x1 + patch_size
        x2 = patch_size*(patch_influenced//l)
        y2 = x2+patch_size
        img[:,x2:y2,x1:y1] = img[:,x2:y2,x1:y1]/count[patch_influenced]
    return img.int()

  else: #max influencer option
    #extract max influencer over list of influence
    if field:
      fields = {}
    max_influencer = defaultdict()

    for influencer in list_of_influence.keys():
      for influenced in list_of_influence[influencer]:
        
        if (influenced[0] not in list(max_influencer.keys()) or max_influencer[influenced[0]][1]<influenced[1]):
          max_influencer[influenced[0]] = [influencer,influenced[1]]
    #drawing influiencer patchs
    for j in range(len(keys)):
      #drawing influencer path
      key = keys[j] 
      
      x1 = patch_size*(key%l)
      y1 = x1 + patch_size
      x2 = patch_size*(key//l)
      y2 = x2+patch_size
      img[:,x2:y2,x1:y1] = patch_gen_influencer(colors[j],patch_size)
    for influenced in list(max_influencer.keys()):
      if influenced not in keys:
        x1 = patch_size*(influenced%l)
        y1 = x1 + patch_size
        x2 = patch_size*(influenced//l)
        y2 = x2+patch_size
        influencer = max_influencer[influenced][0] #extracting influencer patch
        if influencer in keys:
          img[:,x2:y2,x1:y1] = patch_gen(key_col[influencer],patch_size)
          if field:
            if influencer not in list(fields.keys()):
              fields[influencer]=[]
            fields[influencer].append(influenced)
    if field:
      return img.int(),fields
    return img.int()
        


def self_att(image_att,depth,n,mix,colors,n_heads,num_patch,field=False):


  depth_att = (torch.sum(torch.stack([image_att[depth,head] for head in range(n_heads)]),dim=0)/n_heads)
  tmp = torch.stack([torch.sum(depth_att[:,j]) for j in range(num_patch)]) #sum on column is how much a patchg is important
  tmp = (tmp>torch.quantile(tmp,0.75)).int()
  most_importants = torch.nonzero(tmp).squeeze(dim=1)

  list_of_most_importants = torch.stack([torch.stack(torch.sort(torch.sum(torch.stack([image_att[depth,h,patch,:] for h in range(n_heads)])/n_heads,dim=0),descending=True))[:n,:n].T for patch in range(num_patch)]) #returns values and indexes
  lomi_without_values = torch.stack([torch.argsort(torch.sum(torch.stack([image_att[depth,h,patch,:] for h in range(n_heads)])/n_heads,dim=0),descending=True)[:5] for patch in range(num_patch)])
  #to do update most_importants

  list_of_influence = defaultdict(list)

  for i in range(len(lomi_without_values)): #
    #if i in most_importants: #to compute only for most importants
      for j in list_of_most_importants[i]:
        if j[1] in most_importants:
          list_of_influence[int(j[1])].append([i,float(j[0])]) #add patch that is influenced and att value
  
  if field:
    drawed_att, field_of_attention = draw_att(list_of_influence,colors,mix=mix,field=field)
  else:
    drawed_att = draw_att(list_of_influence,colors,mix=mix,field=field)

  attention_for_visualize = torch.einsum('chw -> hwc',drawed_att)

  if field:
    return attention_for_visualize,field_of_attention
  return(attention_for_visualize)

def print_self_att(img,model,colors,depth=0,n=8,mix=False):
  img_batch = img.to('cuda').unsqueeze(dim=0)
  decoder_rel_atts, loss,y,mask = model(img_batch,mask_ratio=0.75,rel_att=True,grid_idx = 0)

  att_map = self_att(decoder_rel_atts[0],depth,n,mix,colors)


  figure = plt.figure(figsize=(16, 16))
  printable = to_print(img.detach().cpu(),inet_std,inet_mean,show=False)

  figure.add_subplot(1, 3, 1)
  plt.title('Image')
  plt.axis("off")
  plt.imshow(printable) #could need to use img.squeeze() to remove any dimension equal to 1 on the input

  figure.add_subplot(1, 3, 2)
  plt.title('Self-attention')
  plt.axis("off")
  plt.imshow(att_map) #could need to use img.squeeze() to remove any dimension equal to 1 on the input

  plt.show()
    
def importance_over_mask(img,model,depth,grid_idx = None):

  img_batch = img.to('cuda').unsqueeze(dim=0)
  decoder_rel_att, loss,y,mask = model(img_batch,mask_ratio=0.75,rel_att=True,grid_idx= grid_idx)
  image_att = decoder_rel_att[0]

  depth_att = (torch.sum(torch.stack([image_att[depth,head] for head in range(n_heads)]),dim=0)/n_heads)
  tmp = torch.stack([torch.sum(depth_att[:,j]) for j in range(num_patch)]) #sum on column is how much a patchg is important
  tmp = (tmp>torch.quantile(tmp,0.75)).int()
  most_importants = torch.nonzero(tmp).squeeze(dim=1)

  mask_bis = torch.nonzero(1-mask[0]).squeeze(dim=1)
  tmp, counts = torch.cat([mask_bis, most_importants]).unique(return_counts=True)
  intersection = tmp[torch.where(counts.gt(1))]
  return(len(intersection)/len(most_importants))

def consistency(decoder_rel_att,mask,num_patch,n_heads,depth): #same function different inputs
  image_att = decoder_rel_att[0]

  depth_att = (torch.sum(torch.stack([image_att[depth,head] for head in range(n_heads)]),dim=0)/n_heads)
  tmp = torch.stack([torch.sum(depth_att[:,j]) for j in range(num_patch)]) #sum on column is how much a patchg is important
  tmp = (tmp>torch.quantile(tmp,0.75)).int()
  most_importants = torch.nonzero(tmp).squeeze(dim=1)

  mask_bis = torch.nonzero(1-mask[0]).squeeze(dim=1)
  tmp, counts = torch.cat([mask_bis, most_importants]).unique(return_counts=True)
  intersection = tmp[torch.where(counts.gt(1))]
  return(len(intersection)/len(most_importants))

def dist(a,b,p=8):
  x_a = a%p #p is number of patch per line
  y_a = a//p

  x_b = b%p
  y_b = b//p

  return(((y_b-y_a)**2+(x_b-x_a)**2)**(1/2))