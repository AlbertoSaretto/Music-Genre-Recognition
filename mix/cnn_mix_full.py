from utils_mix import NNET1D, LN1D, NNET2, LN2D, DataAudio_double
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
from torch.utils.data import Dataset
import warnings
import librosa
from utils_mgr import getAudio


"""
Insired by this tutorial:
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html


IMPORTANT THINGS TO TAKE INTO ACCOUNT:

1. PARAMETERS OF CNNs ARE NOT TRAINED. ONLY THE LAST LAYER IS TRAINED. TO DO THIS WE NEED TO FREEZE THE WEIGHTS OF THE CNNs.
    TO DO SO WE NEED TO SET param.requires_grad = False FOR EACH PARAMETER OF THE CNNs.
        (taken from the tutorial)
        model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        for param in model_conv.parameters():
            param.requires_grad = False


2. SET OPTIMIZER PROPERLY AS SUCH:
    (taken from the tutorial)
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

"""

# Pay attention: this is NOT the same function used in utils_mgr.py
# Here we don't specify architecture type since both are used
def import_and_preprocess_data(PATH_DATA = "../."):

    os.chdir(PATH_DATA)
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
    #test_set  = meta_subset[meta_subset["split"] == "test"]

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
     
    train_dataset    = DataAudio_double(train_set, transform = transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=os.cpu_count())
    
    val_dataset      = DataAudio_double(val_set, transform = transforms)
    val_dataloader   = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count())
    
    #test_dataset     = DataAudio_double(test_set, transform = transforms)
    #test_dataloader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=os.cpu_count())
    

    #return train_dataloader, val_dataloader, test_dataloader
    return train_dataloader, val_dataloader
    

def build_convolutional_blocks(nnet1d, nnet2d):
    
    # Get all convolutional layers from nnet2d
    conv_layers_2d = [layer for layer in nnet2d.modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.ReLU)]

    # Build a new convolutional layer
    conv_block2D = nn.Sequential(*conv_layers_2d[:9]) # [:9] to remove redundat ReLU layers taken by mistake from fc layers
        
    # Get all convolutional layers from nnet1d
    conv_layers_1d = [layer for layer in nnet1d.modules() if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.ReLU)]

    # Build a new convolutional layer
    conv_block1D = nn.Sequential(*conv_layers_1d[:12])


    return conv_block1D, conv_block2D


class MixNet(nn.Module):
    def __init__(self, conv_block1D, conv_block2D):
        super(MixNet, self).__init__()
        self.conv_block1D = conv_block1D
        self.conv_block2D = conv_block2D

        self.dropout = nn.Dropout(p=0.5)  # Add dropout layer

        self.classifier = nn.Sequential(
            nn.Linear(512+2048, 128),
            nn.ReLU(),
            self.dropout,   
            nn.Linear(128, 8),
            nn.Softmax(dim=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Initialize only self.classifer weights
        # We need the weights of the trained CNNs
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
        
        

    def forward(self, x):
        audio = x[0]
        mel = x[1]
        
        conv2d = self.conv_block2D(mel)
        max_pool = F.max_pool2d(conv2d, kernel_size=(125,1))
        avg_pool = F.avg_pool2d(conv2d, kernel_size=(125,1))
        cat2d = torch.cat([max_pool,avg_pool],dim=1)
        cat2d = cat2d.view(cat2d.size(0), -1) # cat2d shape torch.Size([1, 512])
        
        conv1d = self.conv_block1D(audio)
        max_pool = F.max_pool1d(conv1d, kernel_size=125)
        avg_pool = F.avg_pool1d(conv1d, kernel_size=125)
        cat1d = torch.cat([max_pool,avg_pool],dim=1)
        cat1d = cat1d.view(cat1d.size(0), -1) # cat1d dim = torch.Size([batch_size, 2048])

        # Concatanate the two outputs and pass it to the classifier
        # cat1d dim = torch.Size([batch_size, 2048])
        # cat2d dim = torch.Size([batch_size, 512])
        x = torch.cat([cat1d, cat2d], dim=1) 
        x = self.dropout(x)  # Add dropout layer
        x = self.classifier(x)
        return x


class LitMixNet(pl.LightningModule):
    
    def __init__(self, conv_block1D, conv_block2D,config=None):
        super().__init__()
        
        print("LitMixNet initialized")

        self.net = MixNet(conv_block1D, conv_block2D)
        self.best_val = np.inf

        # If no configurations regarding the optimizer are specified, use the default ones
        try:
            self.optimizer = Adadelta(self.net.classifier.parameters(),
                                       lr=config["lr"],rho=config["rho"], eps=config["eps"], weight_decay=config["weight_decay"])
            print("loaded parameters from pickle")
            print("optimzier parameters:", self.optimizer)
        except:
                print("Using default optimizer parameters")
                self.optimizer = Adadelta(self.net.classifier.parameters())
                print("optimzier parameters:", self.optimizer)
        
    def forward(self, x):
        return self.net(x)

    
    # Training_step defines the training loop. 
    def training_step(self, batch, batch_idx=None):
        # training_step defines the train loop. It is independent of forward

        """
        Batch is now composed of three elements: audio, mel and label
        audio and mel are the inputs, label is the target
        """
        x_batch = batch[:2]
        label_batch = batch[2]
        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch) 
        return loss

    def validation_step(self, batch, batch_idx=None):
        # validation_step defines the validation loop. It is independent of forward
        # When the validation_step() is called,
        # the model has been put in eval mode and PyTorch gradients have been disabled. 
        # At the end of validation, the model goes back to training mode and gradients are enabled.
        """
        Batch is now composed of three elements: audio, mel and label
        audio and mel are the inputs, label is the target
        """
        x_batch = batch[:2]
        label_batch = batch[2]
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
        """
        Batch is now composed of three elements: audio, mel and label
        audio and mel are the inputs, label is the target
        """
        # this is the test loop
        x_batch = batch[:2]
        label_batch = batch[2]
        out = self.net(x_batch)
        loss = F.cross_entropy(out, label_batch)

        test_acc = np.sum(np.argmax(label_batch.detach().cpu().numpy(), axis=1) == np.argmax(out.detach().cpu().numpy(), axis=1)) / len(label_batch)
        # Saves lighting_logs to track test loss and accuracy. 
        self.log("test_loss", loss.item(), prog_bar=True)
        self.log("test_acc", test_acc, prog_bar=True)

    def configure_optimizers(self):
        return self.optimizer


# Maybe this function could be put in utils_mgr?

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

    """
    In the first part of the main function we load the weights of the two CNNs and
    build the convolutional blocks that will be used in the MixNet.

    """
    # Load model weights from checkpoint
    CKPT_PATH_1D = "./1Dcheckpoint_Albi.ckpt"
    CKPT_PATH_2D = "../lightning_logs_2D_final/version_4/checkpoints/epoch=69-step=7000.ckpt"
    nnet1d = LN1D.load_from_checkpoint(checkpoint_path=CKPT_PATH_1D).eval()
    nnet2d = LN2D.load_from_checkpoint(checkpoint_path=CKPT_PATH_2D).eval()

    """    # Freeze the weights
    for param in nnet1d.parameters():
        param.requires_grad = False
        
    for param in nnet2d.parameters():
        param.requires_grad = False
    """

    # Build convolutional blocks
    conv_block1D, conv_block2D = build_convolutional_blocks(nnet1d, nnet2d)

    # Build the model

    # Uncomment the following to load Optuna hyperparameters
    #hyperparameters = load_optuna("./trialv2.pickle")
    #model = LitNet(hyperparameters)
    
    # Comment this if you want to load params with Optuna
    model = LitMixNet(conv_block1D, conv_block2D)

    
    # Load model weights from checkpoint
    # Comment/uncomment this three lines
    #CKPT_PATH = "./lightning_logs/version_4/checkpoints/epoch=69-step=7000.ckpt"
    #checkpoint = torch.load(CKPT_PATH)
    #model.load_state_dict(checkpoint['state_dict'])

    """
    Here we define the Trainer and start the training.
    """
    
    # Define the EarlyStopping callback
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # Monitor the validation loss
        min_delta=0.01,     # Minimum change in the monitored metric
        patience=20,          # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'           # Mode: 'min' if you want to minimize the monitored quantity (e.g., loss)
    )

    
    """
    Trainer can start from already existing checkpoint, but we find it more practical to comment/uncomment the lines below
    (see CKPT_PATH part).

    Use trainer to set number of epochs and callbacks.
    """
    trainer = pl.Trainer(max_epochs=100, check_val_every_n_epoch=5, log_every_n_steps=1, 
                         deterministic=True,callbacks=[early_stop_callback],
                           ) # profiler="simple" add this to check where time is spent
    

    

    #train_dataloader, val_dataloader, test_dataloader = import_and_preprocess_data(PATH_DATA = "../.")
    train_dataloader, val_dataloader = import_and_preprocess_data(PATH_DATA = "../.")
    #print("data shape",train_dataloader.dataset.__getitem__(0)[0].shape)
    trainer.fit(model, train_dataloader, val_dataloader)
    #trainer.test(model=model,dataloaders=test_dataloader,verbose=True)


if __name__ == "__main__":
    main()