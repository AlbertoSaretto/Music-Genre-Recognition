import numpy as np
from torchvision.transforms import Compose, ToTensor
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

from utils_mgr import readheavy, get_stft, clip_stft, DataAudio

print("let's start")

##tensorboard --logdir=lightning_logs/ 
# to visualize logs

class NNET2(nn.Module):
        
    def __init__(self):
        super(NNET2, self).__init__()
        
        
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256,kernel_size=(4,513)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(2,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(4, 1),padding=(1,0)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 300),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(150, 8),
            nn.Softmax(dim=1)
        )

        #self.apply(self._init_weights)
    """
    def _init_weights(self, module):
    
        if model_weights is None:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
        else:
            ???
    """

    def forward(self,x):
        
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        x = c1 + c3
        max_pool = F.max_pool2d(x, kernel_size=(125,1))
        avg_pool = F.avg_pool2d(x, kernel_size=(125,1))
        x = max_pool + avg_pool
        x = self.fc(x.view(x.size(0), -1))
        return x 


# Define a LightningModule (nn.Module subclass)
# A LightningModule defines a full system (ie: a GAN, autoencoder, BERT or a simple Image Classifier).
class LitNet(pl.LightningModule):
    
    def __init__(self, config: dict):
       
        super().__init__()
        # super(NNET2, self).__init__() ? 
        
        print('Network initialized')
        
        self.net = NNET2()
        self.val_loss = []
        self.train_loss = []
        self.best_val = np.inf
        self.config = config
        

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
        optimizer = Adadelta(self.net.parameters(), lr=self.config["lr"],rho=self.config["rho"], eps=self.config["eps"], weight_decay=self.config["weight_decay"])
        return optimizer
    

def import_and_preprocess_data(config: dict, test = False,n_train=1):
    
    if test:
        test = readheavy("test",2,"Data/Audio/")
        test = get_stft(test, get_log = True)
        test_clip = clip_stft(test, 128)
        transforms = Compose([ ToTensor(), ])
        test_dataset = DataAudio(data=test_clip,transform=transforms)
        # Qui imposto una batch size arbitraria. Lo faccio perché  temo che la funzione di main mi dia problemi
        test_dataloader = DataLoader(test_dataset, 64, shuffle=False, num_workers=os.cpu_count())
        return test_dataloader
    print("reading train")
    #Convert Audio into stft data
    #train = readheavy("training",1,f"Audio/")
    
    train =  np.load(f"Data/Audio/training_{n_train}.npy", allow_pickle = True)
    
    print("getting stft")
    train_stft = get_stft(train, get_log=True)
   
    del train
    gc.collect()

    valid = readheavy("validation",2,"Data/Audio/")
    valid_stft = get_stft(valid, get_log=True)

    del valid
    gc.collect()

    # take each song and splits it into clips of n_samples 
    # creates 
    print("making clips")
    train_clip = clip_stft(train_stft, 128)
    print("making clips")
    valid_clip = clip_stft(valid_stft, 128)

    del train_stft
    del valid_stft

    gc.collect()

    transforms = Compose([ ToTensor(), ]) # Normalize(0,1) is not necessary for stft data

    train_dataset = DataAudio(data=train_clip,transform=transforms)
    valid_dataset = DataAudio(data=valid_clip,transform=transforms)

    del train_clip
    del valid_clip

    gc.collect()

    #Creation of dataloader classes
    batch_size = config["batch_size"]
    print("train dataloader")
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=os.cpu_count())
    print("valid dataloader")
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=os.cpu_count())
    print("data loaded")

    del train_dataset
    del valid_dataset
    gc.collect()

    return train_dataloader, valid_dataloader




def main(test=False,n_train=1):
    pl.seed_everything(666)
  
    # Set the hyperparameters in the config dictionary
    # Parameters found with Optuna. Find a way to automatically import this

    # Specify the path to the pickle file
    file_path = "./trial.pickle"

    # Open the pickle file in read mode
    with open(file_path, "rb") as file:
        # Load the data from the pickle file
        best_optuna = pickle.load(file)
    
    best_optuna.params["lr"] = 0.01 #changing learning rate
    hyperparameters = best_optuna.params
    
    # Define the EarlyStopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        min_delta=0.01,     # Minimum change in the monitored metric
        patience=3,          # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'           # Mode: 'min' if you want to minimize the monitored quantity (e.g., loss)
    )


    # I think that Trainer automatically takes last checkpoint.
    trainer = pl.Trainer(max_epochs=100, check_val_every_n_epoch=5, log_every_n_steps=10, deterministic=True,callbacks=[early_stop_callback], )
    model = LitNet(hyperparameters)

    # Load model weights from checkpoint
    #CKPT_PATH = "./lightning_logs/version_6/checkpoints/epoch=24-step=2975.ckpt"
    #checkpoint = torch.load(CKPT_PATH)
    #model.load_state_dict(checkpoint['state_dict'])

    if test==False:
        train_dataloader, val_dataloader = import_and_preprocess_data(config=hyperparameters, n_train=n_train)
        trainer.fit(model, train_dataloader, val_dataloader)

    elif test==True:
        test_dataloader = import_and_preprocess_data(config=hyperparameters,test=True)
        trainer.test(model=model,dataloaders=test_dataloader,verbose=True)


if __name__ == "__main__":
    main(test=False,n_train=5)