import numpy as np
import torchvision.transforms.v2 as v2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os 
from torch.optim import Adadelta
from tqdm import tqdm
import gc
import optuna
import pytorch_lightning as pl
import time
import pickle
import utils
from utils_mgr import DataAudio, create_subset, DataAudioH5
import os
import h5py

"""
This script requires h5 files. To create them, run create_h5_from_npy.py
The files should look like this: train_x.h5, train_y.h5, valid_x.h5, valid_y.h5, test_x.h5, test_y.h5
They are loaded in DataAudioH5 class.
Set the correct folder in the main function (dataset_folder)
"""

print("let's start")

##tensorboard --logdir=lightning_logs/ 
# to visualize logs

def import_and_preprocess_data(architecture_type="1D",dataset_folder="./h5_experimentation/"):
    
    
    # Standard transformations for images
    # Mean and std are computed on one file of the training set
    transforms = v2.Compose([v2.ToTensor(),
        v2.RandomResizedCrop(size=(128,513), antialias=True), 
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[1.0784853], std=[4.0071154]),
        ])

    # Create the datasets and the dataloaders
    x_train = h5py.File("./h5_experimentation/train_x.h5", "r")["train"][:]
    y_train = h5py.File("./h5_experimentation/train_y.h5", "r")["train"][:]

    train_dataset    = DataAudioH5(dataset_folder=dataset_folder, dataset_type="train", transform=transforms,input_type=architecture_type)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count())

    val_dataset      = DataAudioH5(dataset_folder=dataset_folder, dataset_type="valid", transform=transforms,input_type=architecture_type)
    val_dataloader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count())

    test_dataset     = DataAudioH5(dataset_folder=dataset_folder, dataset_type="test", transform=transforms,input_type=architecture_type)
    test_dataloader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count())


    return train_dataloader, val_dataloader, test_dataloader





class NNET2(nn.Module):
        
    def __init__(self,initialisation="xavier"):
        super(NNET2, self).__init__()
        
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256,kernel_size=(4,513)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(.2)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(2,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(.2)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(1,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(.2)
        )
               

        self.fc = nn.Sequential(
            nn.Linear(512, 300),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(150, 8),
            nn.Softmax(dim=1)
        )
    """
       if self.initialisation == "xavier":
            self.reset_parameters()

        elif self.initialisation == "model_parameters":
            qui voglio assicurarmi che se non ho un modello salvato, allora lo inizializzo con xavier
            altrimenti uso i parametri del modello salvato

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    """
    def forward(self,x):
        
        c1 = self.c1(x) 
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        x = c1 + c3
        max_pool = F.max_pool2d(x, kernel_size=(125,1))
        avg_pool = F.avg_pool2d(x, kernel_size=(125,1))
        x = torch.cat([max_pool,avg_pool],dim=1)
        x = self.fc(x.view(x.size(0), -1)) # maybe I should use flatten instead of view
        return x 



# Define a LightningModule (nn.Module subclass)
# A LightningModule defines a full system (ie: a GAN, autoencoder, BERT or a simple Image Classifier).
class LitNet(pl.LightningModule):
    
    def __init__(self, config=None):
       
        super().__init__()
        # super(NNET2, self).__init__() ? 
        
        print('Network initialized')
        
        self.net = NNET2()
        self.val_loss = []
        self.train_loss = []
        self.best_val = np.inf
        
        # If no configurations regarding the optimizer are specified, use the default ones
        try:
            self.optimizer = Adadelta(self.net.parameters(),
                                       lr=config["lr"],rho=config["rho"], eps=config["eps"], weight_decay=config["weight_decay"])
            print("loaded parameters from pickle")
            print("optimzier parameters:", self.optimizer)
        except:
                print("Using default optimizer parameters")
                self.optimizer = Adadelta(self.net.parameters())
                print("optimzier parameters:", self.optimizer)
        

    def forward(self,x):
        return self.net(x)

    # Training_step defines the training loop. 
    def training_step(self, batch, batch_idx=None):
        # training_step defines the train loop. It is independent of forward
        x_batch = batch[0]
        label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch) # Diego nota: aggiungere weights in base a distribuzione classi dataset?
        self.train_loss.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx=None):
        # validation_step defines the validation loop. It is independent of forward
        # When the validation_step() is called,
        # the model has been put in eval mode and PyTorch gradients have been disabled. 
        # At the end of validation, the model goes back to training mode and gradients are enabled.
        x_batch = batch[0]
        label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch)

        val_acc = np.sum(np.argmax(label_batch.detach().cpu().numpy(), axis=1) == np.argmax(out.detach().cpu().numpy(), axis=1)) / len(label_batch)

            
        #validation_loss = np.append(validation_loss,1 - val_acc)
        #print(f"accuracy: {val_acc*100} %")
        
        # Should I save the model based on loss or on accuracy?7
        """
        LitNet doesnt need to save the model, it is done automatically by pytorch lightning
        if loss.item() < self.best_val:
            print("Saved Model")
            torch.save(self.net.state_dict(), "saved_models/nnet2/model.pt")
            self.best_val = loss.item()
        """


        self.val_loss.append(loss.item())
        self.log("val_loss", loss.item(), prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)


    def test_step(self, batch, batch_idx):
        # this is the test loop
        x_batch = batch[0]
        label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch)

        test_acc = np.sum(np.argmax(label_batch.detach().cpu().numpy(), axis=1) == np.argmax(out.detach().cpu().numpy(), axis=1)) / len(label_batch)

        self.log("test_loss", loss.item(), prog_bar=True)
        self.log("test_acc", test_acc, prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer
    

def load_optuna( file_path = "./trial.pickle"):
    # Specify the path to the pickle file


    # Open the pickle file in read mode
    with open(file_path, "rb") as file:
        # Load the data from the pickle file
        best_optuna = pickle.load(file)
    
    best_optuna.params["lr"] = 0.5 #changing learning rate
    hyperparameters = best_optuna.params
    
    return hyperparameters


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
    
    hyperparameters = load_optuna("./trialv2.pickle")
    model = LitNet(hyperparameters)
    
    #model = LitNet()

    """
    # Load model weights from checkpoint
    CKPT_PATH = "./lightning_logs/version_1/checkpoints/epoch=29-step=3570.ckpt"
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    """

    train_dataloader, val_dataloader, test_dataloader = import_and_preprocess_data(architecture_type="2D",
                                                                                   dataset_folder="./h5_experimentation/")
    
    
    print("data shape",train_dataloader.dataset.__getitem__(0)[0].shape)
    print("train length",train_dataloader.dataset.__len__())
    print("val length",val_dataloader.dataset.__len__())
    print("test length",test_dataloader.dataset.__len__())
    
    
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model=model,dataloaders=test_dataloader,verbose=True)


if __name__ == "__main__":
    main()