import os
import pandas as pd
import torch
from torch.utils.data import Dataset 
import cv2
from PIL import Image
import numpy as np
from PIL import ImageFile
import csv
import  tqdm
 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from IPython.display import display


#THIS IS THE DATALOADER
class data_set(Dataset):
    def __init__(self , csv_file  , root_dir , transform = None ):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        
        
    def __len__(self):
        return len(self.annotations) 
    
    
    def __getitem__(self , index):
        try:
            img_path = os.path.join(self.root_dir , self.annotations.iloc[index , 0])
            image = Image.open(img_path )

            if self.transform:
                image = self.transform(image)

      
        except:
            return None

        return image

batch_size = 1
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


ImageFile.LOAD_TRUNCATED_IMAGES = True

my_transforms = transforms.Compose([
    #transforms.ToPILImage() , 
    transforms.Resize((64,64)) , 
    transforms.ToTensor() , 
    transforms.Normalize((0.0 , 0.0 , 0.0), (1.0 , 1.0 , 1.0)),
])

data=[]
i = 0
with open('image_annotation.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    
    
    for filename in os.listdir(r"C:\Users\Abhrant\Desktop\abhrant\work\DEEP_LEARNING\FaceDataset"):
        
            data.append(filename)
            writer.writerow(data)
            data=[]
            i += 1
            if i >= 23000:
                break
writeFile.close()

colored_dataset = data_set(csv_file = "image_annotation.csv" , root_dir = r"C:\Users\Abhrant\Desktop\abhrant\work\DEEP_LEARNING\FaceDataset" , 
                          transform = my_transforms)



image_loader = DataLoader(dataset = colored_dataset , collate_fn = collate_fn ,  batch_size = batch_size , shuffle = False)


#THIS IS THE BEGINNING OF THE ARCHITECTURE

#THESE ARE THE FLATTEN AND UNFLATTEN CLASSES
class Flatten(nn.Module):
    def forward(self , input):
        return input.view(input.size(0) , -1)
    
class Unflatten(nn.Module):
    def __init__(self , channel , height , width):
        super(Unflatten , self).__init__()
        
        self.channel = channel
        self.height = height
        self.width = width
          
    def forward(self , input):
        return input.view(input.size(0) , self.channel , self.height , self.width)



#HERE THE WEIGHTS OF THE VARIOUS LAYERS OF THE NETWORK HAVE BEEN INITIALIZED
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.uniform_(m.weight, -0.08 , 0.08)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.uniform_(m.weight, -0.08 , 0.08)
        torch.nn.init.zeros_(m.bias)
    if classname.find('ConvTranspose') != -1:
        torch.nn.init.uniform_(m.weight, -0.08 , 0.08)
        

#THIS IS THE MAIN ARCHITECTURE OF THE NETWORK
class convVAE(nn.Module):
    
    def __init__(self , latent_size):
        super(convVAE , self).__init__()
        
        self.latent_size = latent_size
        
        self.encoder = nn.Sequential(
             
            nn.Conv2d(3 , 32 , 3 , 1 , 1) ,
            nn.MaxPool2d(2 , 2) , 
            nn.ReLU() ,
            nn.BatchNorm2d(32) ,
             
            
            nn.Conv2d(32 , 64 , 3 , 1 , 1) , 
            nn.MaxPool2d(2 , 2) , 
            nn.ReLU() ,
            nn.BatchNorm2d(64) ,
            
            nn.Conv2d(64 , 128 , 3 , 1 , 1) , 
            nn.MaxPool2d(2 , 2) ,
            nn.ReLU() ,
            nn.BatchNorm2d(128) , 
            
            nn.Conv2d(128 , 256 , 3 , 1 , 1) , 
            nn.MaxPool2d(2 , 2) , 
            nn.ReLU() , 
            nn.BatchNorm2d(256) , 
            
            Flatten() , 
            nn.Linear(4096 , 1024) ,
            nn.Linear(1024 , 32) ,
            nn.ReLU()            
            )
        
        self.mu = nn.Linear(32 , self.latent_size)
        self.logvar = nn.Linear(32 , self.latent_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size , 1024) , 
            nn.ReLU() , 
            nn.Linear(1024 , 4096) , 
            nn.ReLU() , 
            Unflatten(256 , 4 , 4) , 
           
            nn.ConvTranspose2d(256 , 128 , 2 , 2) , 
            nn.ReLU() , 
            nn.BatchNorm2d(128) , 
            
            nn.ConvTranspose2d(128 , 64 , 2 , 2) , 
            nn.ReLU() , 
            nn.BatchNorm2d(64) ,
            
            nn.ConvTranspose2d(64 , 32 , 2 , 2) , 
            nn.ReLU() , 
            nn.BatchNorm2d(32) ,
            
            nn.ConvTranspose2d(32 , 3 , 2 , 2) , 
            nn.Sigmoid()
            )
        
    def encode(self , x):
        h = self.encoder(x)
        mu , logvar = self.mu(h) , self.logvar(h)
        return mu , logvar            
        
    def reparameterize(self , mu , logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu +eps * std
        return z
        
    def decode(self , z):
        decoded_z = self.decoder(z)
        return decoded_z
        
    def forward(self , x):
        mu , logvar = self.encode(x)
        z = self.reparameterize(mu , logvar)
        return self.decode(z) , mu , logvar
            
vae = convVAE(32)
vae.to("cuda")
vae.apply(weights_init)


#HERE THE LOSS FUNCTION FO THE NETWORK IS DEFINED
def loss_function(recon_x , x , mu , logvar):
    BCE = F.binary_cross_entropy(recon_x , x , reduction = "sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return torch.mean(BCE + KLD)

#INSTEAD OF USING RMSPROP, OTHEEOPTIMIZERS LIKE ADAM CAN ALSO BE USED
optimizer = optim.RMSprop(vae.parameters() , lr = 0.001)

#THIS IS THE TRAINING LOOP OF THE NETWORK
EPOCHS = 200

loss_list = []
epoch_list = []

for epoch in range(EPOCHS):
    epoch_list.append(epoch)
    train_loss = 0
    
    for i in tqdm(image_loader):
            input_img = i.to("cuda")

            optimizer.zero_grad()
            output , mu , logvar = vae.forward(input_img)

            loss = loss_function(output , input_img , mu , logvar)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() 
            input_img.to("cpu")
          
       
    train_loss = train_loss / 23000
    loss_list.append(train_loss)
    print(epoch , train_loss)

#THE WIEGHT FILE WILL BE SAVED IN THE GIVEN LOCATION AND CAN BE LATER ACCESSED TO RECOSTRUCT IMAGES OR GENERATE NEW IMAGES.
torch.save(vae.state_dict() , r"C:\Users\Abhrant\Desktop\abhrant\work\DEEP_LEARNING\vae_weight.pt")
plt.plot(epoch_list , loss_list)
plt.show()
