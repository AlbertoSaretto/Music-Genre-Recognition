import numpy as np
import torchvision.transforms.v2 as v2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os 
from torch.optim import Adadelta
import pytorch_lightning as pl
import pickle
import utils
from utils_mgr import DataAudio, create_subset, MinMaxScaler
import os


print("let's start")

##tensorboard --logdir=lightning_logs/ 
# to visualize logs

def import_and_preprocess_data(architecture_type="1D"):


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

        # Weights initialisation
        # if
        if initialisation == "xavier":
            print("initialising weights wit Xavier")
            self.apply(self._init_weights)
        else:
            print('Weights not initialised. If previous checkpoint is not loaded, set initialisation = "xavier"')


    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self,x):
        
        c1 = self.c1(x) 
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        x = c1 + c3
        max_pool = F.max_pool2d(x, kernel_size=(125,1))
        avg_pool = F.avg_pool2d(x, kernel_size=(125,1))
        x = torch.cat([max_pool,avg_pool],dim=1)
        x = self.fc(x.view(x.size(0), -1)) # Reshape x to fit in linear layers. Equivalent to F.Flatten
        return x 



# Define a LightningModule (nn.Module subclass)
# A LightningModule defines a full system (ie: a GAN, autoencoder, BERT or a simple Image Classifier).
class LitNet(pl.LightningModule):
    
    def __init__(self, config=None):
       
        super().__init__()       
        print('Network initialized')
        
        self.net = NNET2()
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
        loss = F.cross_entropy(out, label_batch) 
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
        """
        Validation accuracy is computed as follows.
        label_batch are the true labels. They are one-hot encoded (eg [1,0,0,0,0,0,0,0]). 
        out are the predicted labels. They are a 8-dim vector of probabilities.
        argmax checks what is the index with the highest probability. Each index is related to a Music Genre.
        If the indexes are equal the classification is correct.
        
        """
        val_acc = np.sum(np.argmax(label_batch.detach().cpu().numpy(), axis=1) == np.argmax(out.detach().cpu().numpy(), axis=1)) / len(label_batch)

        """
        LitNet doesnt need to save the model, it is done automatically by pytorch lightning
        if loss.item() < self.best_val:
            print("Saved Model")
            torch.save(self.net.state_dict(), "saved_models/nnet2/model.pt")
            self.best_val = loss.item()
        """
        # Saves lighting_logs used in Tensorboard.
        self.log("val_loss", loss.item(), prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)


    def test_step(self, batch, batch_idx):
        # this is the test loop
        x_batch = batch[0]
        label_batch = batch[1]
        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch)

        test_acc = np.sum(np.argmax(label_batch.detach().cpu().numpy(), axis=1) == np.argmax(out.detach().cpu().numpy(), axis=1)) / len(label_batch)
        # Saves lighting_logs to track test loss and accuracy. 
        self.log("test_loss", loss.item(), prog_bar=True)
        self.log("test_acc", test_acc, prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer
    

def load_optuna( file_path = "./trial.pickle",lr=None):

    """
    Use this function to load hyperparameters found with Optuna.
    These should be saved in a pickle file, containing a dictonary with optimizer's parameters.
    This is not the 100% correct way of using Optuna, in fact one should use the .db  file that the Optuna study creates.
    But this works anyway.

    If you want to change the lr to a value other than the one found with Optuna, you can set it in the input of the function.
    """
    # Open the pickle file in read mode
    with open(file_path, "rb") as file:
        # Load the data from the pickle file
        best_optuna = pickle.load(file)
    if lr is not None:
            
        best_optuna.params["lr"] = lr #changing learning rate

    return  best_optuna.params


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
    trainer = pl.Trainer(max_epochs=100, check_val_every_n_epoch=5, log_every_n_steps=1, 
                         deterministic=True,callbacks=[early_stop_callback], ) # profiler="simple" remember to add this and make fun plots
    
    #hyperparameters = load_optuna("./trialv2.pickle")
    #model = LitNet(hyperparameters)
    
    model = LitNet()

    
    # Load model weights from checkpoint
    CKPT_PATH = "./lightning_logs/version_4/checkpoints/epoch=69-step=7000.ckpt"
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    

    train_dataloader, val_dataloader, test_dataloader = import_and_preprocess_data(architecture_type="2D")
    print("data shape",train_dataloader.dataset.__getitem__(0)[0].shape)
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model=model,dataloaders=test_dataloader,verbose=True)


if __name__ == "__main__":
    main()