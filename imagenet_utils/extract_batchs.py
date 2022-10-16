import os
import pickle
import torch
from tqdm import tqdm

def path_init(base_repo):
    repo_train = base_repo + '\\train\\'
    repo_val = base_repo  +'\\val\\'
    if not os.path.exists(repo_train) and not os.path.exists(repo_val):
      os.mkdir(repo_train)
      os.mkdir(repo_val)
    return(repo_train,repo_val)





def extract_batch(path,repo,repo_val,train=True,val=True,device='cuda'):
    counter = 0
    val_counter=0

    if train:

          for j in range(10):
            x = pickle.load(open(path + 'train_data_batch_' + str(1+j),'rb'))
            n = len(x['data'])
            imgs = torch.tensor(x['data'],dtype=torch.float32).to(device).reshape(n,3,64,64) #add to.('cuda') will be faster
            labels = torch.tensor(x['labels'],dtype=torch.uint8).to(device)

            for i in tqdm(range(len(imgs))):
                f_name = repo + str(counter)+".pt"
                img = imgs[i].clone().cpu()/255.

                #img = T_n(img)
                #label = labels[i]
                torch.save(img,f_name) #saving as list list,label double the space needed
                counter+=1

    if val:
        x = pickle.load(open(path + 'val_data','rb'))
        n = len(x['data'])
        imgs = torch.tensor(x['data'],dtype=torch.float32).to('cuda').reshape(n,3,64,64) #add to.('cuda') will be faster

        for i in tqdm(range(len(imgs))):
            f_name = repo_val + str(val_counter)+".pt"
            img = imgs[i].clone().cpu()/255.

            #img = T_n(img)
            #label = labels[i]
            torch.save(img,f_name) #saving as list list,label double the space needed
            val_counter+=1 #50000
    return(counter,val_counter)