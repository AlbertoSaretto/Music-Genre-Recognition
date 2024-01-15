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



"""
This is the Autoencoder network. It will be trained to recostruct spectrograms.
Once trained, the encoder part will be used to extract the latent space representation of the spectrograms.
This will be used as input for the classifier.
This operation needs to be done in another script, where the trained model is loaded and a classifier is trained on top of it.

"""

print("let's start")

##tensorboard --logdir=lightning_logs/ 
# to visualize logs

"""
Remove this. You'll find it in utils_mgr.py
def MinMaxScaler(x):
    Xmin = torch.min(x)
    Xmax = torch.max(x)
    return (x-Xmin)/(Xmax-Xmin)
"""



"""
USE THIS TO LOAD DATA WITH H5 FILE

def import_and_preprocess_data(architecture_type="1D",dataset_folder="./h5_experimentation/"):
    
    
    # Standard transformations for images
    # Mean and std are computed on one file of the training set
    transforms = v2.Compose([v2.ToTensor(),
        v2.RandomResizedCrop(size=(128,513), antialias=True), 
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
       # v2.Normalize(mean=[1.0784853], std=[4.0071154]),
        v2.Lambda(lambda x: MinMaxScaler(x))
        ])

    # Create the datasets and the dataloaders
    train_dataset    = DataAudioH5(dataset_folder=dataset_folder, dataset_type="train", transform=transforms,input_type=architecture_type)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count())

    val_dataset      = DataAudioH5(dataset_folder=dataset_folder, dataset_type="valid", transform=transforms,input_type=architecture_type)
    val_dataloader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count())

    test_dataset     = DataAudioH5(dataset_folder=dataset_folder, dataset_type="test", transform=transforms,input_type=architecture_type)
    test_dataloader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count())


    return train_dataloader, val_dataloader, test_dataloader
"""


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
        #v2.Normalize(mean=[1.0784853], std=[4.0071154]),
        v2.Lambda(lambda x: MinMaxScaler(x)) # see utils_mgr
        ])

    # Create the datasets and the dataloaders
    train_dataset    = DataAudio(train_set, transform = transforms,type=architecture_type)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count())

    val_dataset      = DataAudio(val_set, transform = transforms,type=architecture_type)
    val_dataloader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count())

    test_dataset     = DataAudio(test_set, transform = transforms,type=architecture_type)
    test_dataloader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count())


    return train_dataloader, val_dataloader, test_dataloader


class Encoder(nn.Module):

    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, 
                      stride=2, padding=1),
            nn.ReLU(True),
             nn.BatchNorm2d(8),
             nn.Dropout2d(0.2),
            # Second convolutional layer
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, 
                      stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            # Third convolutional layer
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, 
                      stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(in_features= 32, out_features=64),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(in_features=64, out_features=encoded_space_dim)
        )
        
    def forward(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        #print("encoder shape",x.shape)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        #print("maxpool shape",x.shape)
        # Flatten
        x = self.flatten(x)
        #print("flatten shape",x.shape)
        # # Apply linear layers
        x = self.encoder_lin(x)
        return x
    

class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(in_features=encoded_space_dim, out_features=64),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(in_features=64, out_features= 32),
            nn.ReLU(True)
        )

        ### Unflatten
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 1, 1))

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, 
                               stride=2, output_padding=(1,0)),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
           
            # Second transposed convolution
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, 
                               stride=2, padding=1, output_padding=(1,0)),
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.2),
           
            # Third transposed convolution
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, 
                               stride=2, padding=1, output_padding=(1,0)),
            nn.ReLU(True),
            nn.BatchNorm2d(1),
            nn.Dropout2d(0.2),
           
        )
        

    def forward(self, x):
        #print("input of decoder",x.shape)
        # Apply linear layers
        x = self.decoder_lin(x)
        #print("decoder linear out",x.shape)
        # Unflatten
        x = self.unflatten(x)
        #print("unflattened shape",x.shape)
        
        # Apply upsampling
        x = F.interpolate(x, size=(15,64), mode='nearest')
        #print("interpolate out shape",x.shape)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        #print("decoder conv out",x.shape)   
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x
    
class Autoencoder(nn.Module):
        
        def __init__(self, encoded_space_dim=64):
            super().__init__()
            self.encoder = Encoder(encoded_space_dim)
            self.decoder = Decoder(encoded_space_dim)
            
            self._initialize_weights()

        def _initialize_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                    init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        init.constant_(module.bias, 1)
                elif isinstance(module, nn.Linear):
                    init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        init.constant_(module.bias, 1)
                
        def forward(self, x):
            #print("very input shape",x.shape)
            x = self.encoder(x)
            x = self.decoder(x)
            return x
# Define a LightningModule (nn.Module subclass)
# A LightningModule defines a full system (ie: a GAN, autoencoder, BERT or a simple Image Classifier).
class LitNet(pl.LightningModule):
    
    def __init__(self, config=None):
       
        super().__init__()      
        print('Network initialized')
        
        self.net = Autoencoder()
        
        # If no configurations regarding the optimizer are specified, use the default ones
        try:
            self.optimizer = Adadelta(self.net.parameters(),
                                       lr=config["lr"],rho=config["rho"], eps=config["eps"], weight_decay=config["weight_decay"])
            print("loaded parameters from pickle")
            print("optimzier parameters:", self.optimizer)
        except:
                print("Using default optimizer parameters")
                self.optimizer = Adadelta(self.net.parameters(),lr=0.1)
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
    pl.seed_everything(666)
  
    # Set the hyperparameters in the config dictionary
    # Parameters found with Optuna. Find a way to automatically import this
   
    # Define the EarlyStopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        min_delta=0.01,     # Minimum change in the monitored metric
        patience=20,          # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'           # Mode: 'min' if you want to minimize the monitored quantity (e.g., loss)
    )


    # I think that Trainer automatically takes last checkpoint.
    trainer = pl.Trainer(max_epochs=1, check_val_every_n_epoch=1, log_every_n_steps=1, 
                         deterministic=True,callbacks=[early_stop_callback], ) # profiler="simple" remember to add this and make fun plots
    
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