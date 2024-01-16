import os 
os.chdir("./../")
print("Current working directory is now: ", os.getcwd())
import numpy as np
import torchvision.transforms.v2 as v2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adadelta
from tqdm import tqdm
import gc
import optuna
import pytorch_lightning as pl
import time
import pickle
import torch.nn.init as init


from utils_mgr import DataAudio, create_subset, MinMaxScaler
import utils




def import_and_preprocess_data(architecture_type="2D"):


    """
    This function uses metadata contained in tracks.csv to import mp3 files,
    pass them through DataAudio class and eventually create Dataloaders.  
    
    """
    # Load metadata and features.
    tracks = utils.load('data/fma_metadata/tracks.csv')

    #Select the desired subset among the entire dataset
    sub = 'small'
    raw_subset = tracks[tracks['set', 'subset'] <= sub] 
    
    #Creation of clean subset for the generation of training, test and validation sets
    meta_subset= create_subset(raw_subset)

    # Remove corrupted files
    corrupted = [98565, 98567, 98569, 99134, 108925, 133297]
    meta_subset = meta_subset[~meta_subset['index'].isin(corrupted)]

    #Split between taining, validation and test set according to original FMA split

    train_set = meta_subset[meta_subset["split"] == "training"]
    val_set   = meta_subset[meta_subset["split"] == "validation"]
    test_set  = meta_subset[meta_subset["split"] == "test"]

    # Standard transformations for images

    # There are two ways to normalize data: 
    #   1. Using  v2.Normalize(mean=[1.0784853], std=[4.0071154]). These values are computed with utils_mgr.mean_computer() function.
    #   2. Using v2.Lambda and MinMaxScaler. This function is implemented in utils_mgr and resambles sklearn homonym function.

    transforms = v2.Compose([v2.ToTensor(),
        v2.RandomResizedCrop(size=(128,513), antialias=True), # Data Augmentation
        v2.RandomHorizontalFlip(p=0.5), # Data Augmentation
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[1.0784853], std=[4.0071154]),
        #v2.Lambda(lambda x: MinMaxScaler(x)) # see utils_mgr
        ])

    # Create the datasets and the dataloaders
    
    train_dataset    = DataAudio(train_set, transform = transforms,type=architecture_type)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count())

    val_dataset      = DataAudio(val_set, transform = transforms,type=architecture_type)
    val_dataloader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count())
    
    test_dataset     = DataAudio(test_set, transform = transforms,type=architecture_type)
    test_dataloader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count())


    return train_dataloader, val_dataloader, test_dataloader


# Residual encoder
class ResEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()   

        self.c1 = nn.Sequential(
             nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, 
                                stride=2, padding=1),
                        nn.Sigmoid(),
                        nn.BatchNorm2d(8),
                        nn.Dropout2d(0.2),
            
                        )       

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, 
                                stride=2, padding=0),
                        nn.Sigmoid(),
                        nn.BatchNorm2d(8),
                        nn.Dropout2d(0.2),
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, 
                                stride=2, padding=1),
                        nn.Sigmoid(),
                        nn.BatchNorm2d(8),
                        nn.Dropout2d(0.2),
        )


    def forward(self, x):
            
            c1 = self.c1(x)
            c2 = self.c2(c1)
            c3 = self.c3(c2)
            
            # Residual connection
            avg_pool = nn.AvgPool2d(kernel_size=(4, 4), stride=(4, 4))
            c1 = avg_pool(c1)
            x  = c1 + c3
            
            return x
    
# Residual decoder
class ResDecoder(nn.Module):
    
    def __init__(self):
        super().__init__()   

        self.c1 = nn.Sequential(
             nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, 
                                stride=3, padding=1),
                        nn.Sigmoid(),
                        nn.BatchNorm2d(8),
                        nn.Dropout2d(0.2),
            
                        )       

        self.c2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, 
                                stride=2, padding=0),
                        nn.Sigmoid(),
                        nn.BatchNorm2d(8),
                        nn.Dropout2d(0.2),
        )

        self.c3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, 
                                stride=3, padding=1),
                        nn.Sigmoid(),
                        nn.BatchNorm2d(8),
                        nn.Dropout2d(0.2),
        )


    def forward(self, x):
            
            c1 = self.c1(x)
            c2 = self.c2(c1)
            c3 = self.c3(c2)
            """
            Using interpolate to control the shapes of the tensors
            Since it's a decoder, it would be difficult to use add the first
            residual connection without using interpolate (c1 has to be not too big since it's the output of the encoded space)          """
            # Residual connection
            # Upsampling c1 to match c3 shape
            c1 = F.interpolate(c1, size= (133, 565), mode='bilinear', align_corners=False)
            x  = c1 + c3
            # Upsampling to match original input shape
            x = F.interpolate(x, size= (128,513), mode='bilinear', align_corners=False)
            
            return x
    

# Residual autoencoder
class ResAE(nn.Module):
    
    def __init__(self, encoded_space_dim=32):
        super().__init__()   
        
        self.encoder = ResEncoder()
        self.decoder = ResDecoder()

         ### Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(in_features= 2048, out_features=512),
            nn.Sigmoid(),
            # Second linear layer
            nn.Linear(in_features=512, out_features=encoded_space_dim),
            nn.Sigmoid()
        )

        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(in_features= encoded_space_dim, out_features=512),
            nn.Sigmoid(),
            # Second linear layer
            nn.Linear(in_features=512, out_features=2048),
            nn.Sigmoid()
        )

        
    def forward(self, x):
        
        x = self.encoder(x)
        # Avg pooling for dimensional reduction
        x = F.avg_pool2d(x, kernel_size=(2,2)) # shape torch.Size([batch_size, 8, 8, 32])
        # Flatten
        x = torch.flatten(x, start_dim=1) # shape torch.Size([batch_size, 2048])
        # Linear section
        latent_space = self.encoder_lin(x)
        # Start decoding
        x = self.decoder_lin(latent_space)
        # Unflatten
        x = torch.unflatten(x,1,(8,8,32))
        # Decode
        x = self.decoder(x)
        # Apply sigmoid to force output between 0 and 1
        x = torch.sigmoid(x)
        return x
    

#Define a LightningModule (nn.Module subclass)
# A LightningModule defines a full system (ie: a GAN, autoencoder, BERT or a simple Image Classifier).
class LitNet(pl.LightningModule):
    
    def __init__(self, config=None):
       
        super().__init__()      
        print('Network initialized')
        
        self.net = ResAE()
        
        # If no configurations regarding the optimizer are specified, use the default ones
        try:
            self.optimizer = Adadelta(self.net.parameters(),
                                       lr=config["lr"],rho=config["rho"], eps=config["eps"], weight_decay=config["weight_decay"])
            print("loaded parameters from pickle")
            print("optimzier parameters:", self.optimizer)
        except:
                print("Using default optimizer parameters")
                self.optimizer = Adadelta(self.net.parameters(),lr=0.01)
                print("optimzier parameters:", self.optimizer)
        

    def forward(self,x):
        return self.net(x)

    # Training_step defines the training loop. 
    def training_step(self, batch, batch_idx=None):
        # training_step defines the train loop. It is independent of forward
        x_batch = batch[0]
        #label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.mse_loss(out, x_batch)
        print("loss",loss.item())
        """
        print("loss",loss.item())
        print("x_batch",x_batch,"\n")
        print("out",out,"\n")
        """
        return loss

    def validation_step(self, batch, batch_idx=None):
        # validation_step defines the validation loop. It is independent of forward
        # When the validation_step() is called,
        # the model has been put in eval mode and PyTorch gradients have been disabled. 
        # At the end of validation, the model goes back to training mode and gradients are enabled.
        x_batch = batch[0]
        #label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.mse_loss(out, x_batch)  # Use MSE loss for autoencoders
        """
        print("loss",loss.item())
        print("x_batch",x_batch,"\n")
        print("out",out,"\n")
        """    
        self.log("val_loss", loss.item(), prog_bar=True)


    def test_step(self, batch, batch_idx):
        # this is the test loop
        x_batch = batch[0]
        #label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.mse_loss(out, x_batch)  # Use MSE loss for autoencoders

        self.log("test_loss", loss.item(), prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer
    
def main():
    pl.seed_everything(0)
  
    # Set the hyperparameters in the config dictionary
    # Parameters found with Optuna. Find a way to automatically import this
   
    # Define the EarlyStopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        min_delta=0.01,     # Minimum change in the monitored metric
        patience=5,          # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'           # Mode: 'min' if you want to minimize the monitored quantity (e.g., loss)
    )


    # I think that Trainer automatically takes last checkpoint.
    trainer = pl.Trainer(max_epochs=1, check_val_every_n_epoch=1, log_every_n_steps=1, 
                         deterministic=True,callbacks=[early_stop_callback], 
                         gradient_clip_val=0.5,
                         gradient_clip_algorithm="value") # adding gradient clip to avoid exploding gradients
    # profiler="simple" remember to add this and make fun plots
    
    #hyperparameters = load_optuna("./trialv2.pickle")
    #model = LitNet(hyperparameters)
    
    model = LitNet()

    """
    # Load model weights from checkpoint
    CKPT_PATH = "./lightning_logs/version_1/checkpoints/epoch=29-step=3570.ckpt"
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    """

    train_dataloader, val_dataloader, _ = import_and_preprocess_data(architecture_type="2D")
                                                                                  
    print("data shape",train_dataloader.dataset.__getitem__(0)[0].shape)
    print("everything between 0-1",torch.max(train_dataloader.dataset.__getitem__(0)[0]),
          torch.min(train_dataloader.dataset.__getitem__(0)[0]))
    
    trainer.fit(model, train_dataloader, val_dataloader)
    #trainer.test(model=model,dataloaders=test_dataloader,verbose=True)

    


if __name__ == "__main__":
    main()